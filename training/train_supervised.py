import os
import json

import pyjson5 as json
import torch
import tqdm

import utilities.utils as utils
from utilities import augmentations
import wandb
from models import model_utils
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics.classification import MulticlassAveragePrecision, MulticlassAUROC, MulticlassF1Score
import gc
from timm.scheduler import CosineLRScheduler


class_dict = {0: "Non Deformation", 1: "Deformation"}


def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if (param.grad.norm().item() is None) or (param.grad.norm().item() < 1e-5):
                print(f'Erratic gradients: {name}, {param.grad.norm()}')


def train_epoch(train_loader, model, optimizer, criterion, epoch, configs, verbose, lr_scheduler):
    model.train()

    num_positives, num_negatives = 0, 0
    total_loss = 0.0
    num_samples = 0
    metrics = utils.initialize_metrics(configs)
    for metric in metrics:
        if isinstance(metric, MulticlassF1Score):
            f1score = metric
    
    if configs['mixed_precision']:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()

    # Optimal alternative for optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None

    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.cuda.amp.autocast(enabled=configs['mixed_precision']):
            image, label, sample = batch
            
            image = image.to(configs['device'])
            label = label.to(configs['device']).long()

            if configs['task'] == 'segmentation':
                num_positives += (label.sum(dim=(1, 2)) > 0).sum().item()
                num_negatives += label.shape[0] - (label.sum(dim=(1, 2)) > 0).sum().item()
            elif configs['task'] == 'classification':
                num_positives += (label.sum().item())
                num_negatives += label.shape[0] - (label.sum().item())

            out = model(image)
            predictions = torch.argmax(out,dim=1)

            loss = criterion(input = out, target = label)

            f1score(predictions, label)

            total_loss += loss.item()
            num_samples += image.shape[0]
            if idx%50==0:
                log_dict = {'Epoch':epoch, 'Iteration':idx,'train loss':total_loss/num_samples}
                if configs['wandb']:
                    wandb.log(log_dict)
                elif verbose:
                    print(log_dict)
            if configs['mixed_precision']:
                scaler.scale(loss).backward()
                if configs['gradient_clip'] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), configs['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if configs['gradient_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), configs['gradient_clip'])
                optimizer.step()

            if 'vit' in configs['architecture'].lower():
                if configs['lr_scheduling']:
                    lr_scheduler.step_update(epoch * 2725 + idx)

            check_gradients(model)

    log_dict = {
        'Positives per epoch: ':num_positives,
        'Negatives per epoch: ':num_negatives
    }
    if configs['num_classes'] == 1:
        scores = f1score.compute()
        log_dict['Train ' + f1score.__class__.__name__ + ' Class: Deformation'] = scores
    elif f1score.average != 'none':
        log_dict['Train ' + metric.average + ' ' + f1score.__class__.__name__] = f1score.compute()
    else:
        scores = f1score.compute()
        for idx in range(scores.shape[0]):
            log_dict['Train ' + f1score.__class__.__name__ + ' Class: ' + class_dict[idx]] = scores[idx]

    log_dict['lr'] = optimizer.param_groups[0]['lr']

    if configs['wandb']:
        wandb.log(log_dict)
    elif verbose:
        print('Positives per epoch: ',num_positives)
        print('Negatives per epoch: ',num_negatives)

        if configs['num_classes'] != 1:
            print('F1 score (Non Deformation): ', scores[0])
            print('F1 score (Deformation): ', scores[1])
        else:
            print('F1 score (Deformation): ', scores)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() )


def train(configs, verbose):
    if verbose:
        print('='*20)
        print('Initializing classification trainer')
        print('='*20)

    if configs['wandb']:
        wandb_id = wandb.util.generate_id()

        wandb.init(
            project=configs["wandb_project"],
            entity=configs["wandb_entity"],
            config=configs,
            resume=False,
            id=wandb_id,
            reinit=True
        )

    if configs['wandb']:
        configs['checkpoint_path'] = str(utils.create_checkpoint_directory(configs, wandb_run=wandb.run))
    else:
        configs['checkpoint_path'] = str(utils.create_checkpoint_directory(configs))

    if not configs['wandb']:
        # Save model configs in checkpoint directory
        configs_path = os.path.join(configs['checkpoint_path'],'configs.json')
        augment_path = os.path.join(configs['checkpoint_path'],'augment.json')
        if not os.path.exists(configs_path):
            with open(str(configs_path), 'wb') as f:
                json.dump(configs, f, indent=4)
        if configs["augment"] and not os.path.exists(augment_path):
            with open(str(augment_path),'wb') as f:
                json.dump(augmentations.get_augmentations(configs, 512), f, indent=4)

    criterion = utils.initialize_criterion(configs).to(configs['device'])

    if 'vit' in configs['architecture'].lower():
        model_config = json.load(open('configs/method/vit/vit.json','r'))
        configs.update(model_config)
        if configs['wandb']:
            wandb.config.update(model_config)

    base_model = model_utils.create_model(configs).to(configs['device'])

    if 'vit' in configs['architecture'].lower():
        in_features = base_model.model.head.in_features
    elif 'unet' in configs['architecture'].lower():
        if configs['wandb']:
            unet_config = json.load(open('configs/method/unet/unet.json','r'))
            for key, value in unet_config.items():
                configs['Unet'+key] = value
            wandb.config.update(configs)
    elif 'deeplabv3' in configs['architecture'].lower():
        if configs['wandb']:
            deeplabv3_config = json.load(open('configs/method/deeplabv3/deeplabv3.json','r'))
            for key, value in deeplabv3_config.items():
                configs['deeplabv3'+key] = value
            wandb.config.update(configs)
    elif "segformer" in configs['architecture'].lower():
        if configs['wandb']:
            segformer_config = json.load(open('configs/method/segformer/segformer.json','r'))
            for key, value in segformer_config.items():
                configs['segformer'+key] = value
            wandb.config.update(configs)
    elif 'convnext' in configs['architecture'].lower():
        in_features = base_model.head.fc.in_features
    else:
        if ('mobilenet' not in configs['architecture'].lower()) and ('efficientnet' not in configs['architecture'].lower()):
            in_features = base_model.fc.in_features

    train_loader, val_loader, _ = utils.prepare_supervised_learning_loaders(configs, verbose)

    optimizer = torch.optim.AdamW(base_model.parameters(),lr=configs['lr'],weight_decay=configs['weight_decay'])
    scheduler = None
    if configs['lr_decay']:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    if 'vit' in configs['architecture'].lower():
        if configs['lr_scheduling']:
            steps_per_epoch = 2725  # For 512x512 patch size
            total_steps = configs['epochs'] * steps_per_epoch
            # Cosine decay with warmup
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=total_steps,       # total training steps
                warmup_t=steps_per_epoch * 5,  # number of warm-up epochs
                warmup_lr_init=1e-6,
                lr_min=1e-5,
                t_in_epochs=False,
            )

    model = base_model
    model.to(configs['device'])

    if verbose:
        print('Model Parameters:', count_parameters(model))

    best_loss = 10000.0
    last_best_epoch = 0
    early_stopping = False
    max_epochs = configs['early_stopping'] if configs['early_stopping'] is not None else configs['epochs']
    for epoch in range(configs['epochs']):
        if last_best_epoch<=max_epochs:
            train_epoch(train_loader,model,optimizer,criterion,epoch,configs, verbose, lr_scheduler=scheduler)
            val_loss = test(configs,phase='val',model=model,criterion=criterion,loader=val_loader,epoch=epoch, verbose=verbose)
            if configs['lr_decay'] and 'vit' not in configs['architecture']:
                scheduler.step()
            if val_loss < best_loss:
                last_best_epoch = 0
                best_loss = val_loss
                if verbose:
                    print('New best validation loss: ',best_loss)
                torch.save(base_model,os.path.join(configs['checkpoint_path'],'best_model.pt'))
                if verbose:
                    print('Saving checkpoint')
            else:
                last_best_epoch += 1
        else:
            early_stopping = True
        
    if early_stopping and verbose:
        print(f"Maximum number of epochs with no improvement: {configs['early_stopping']} reached. Early stopping...")

    res = test(configs,phase='test', verbose=verbose)

    wandb.finish(exit_code=0)
    return res


@torch.no_grad()
def test(configs, phase, model=None, loader=None, criterion=None, epoch='Test', verbose=False):
    if phase=='test':
        if verbose:
            print('='*20)
            print('Begin Testing')
            print('='*20)
        _, _, loader = utils.prepare_supervised_learning_loaders(configs, verbose)
        criterion = utils.initialize_criterion(configs).to(configs['device'])
        #Load model from checkpoint
        model = torch.load(os.path.join(configs['checkpoint_path'], 'best_model.pt'), map_location=configs['device'])

    elif phase=='val':
        if verbose:
            print('='*20)
            print('Begin Evaluation')
            print('='*20)
    else:
         print('Uknown phase!')
         exit(3)
    
    metrics = utils.initialize_metrics(configs)

    model.to(configs['device'])
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        with torch.autocast("cuda", enabled=configs['mixed_precision']):
            for idx, batch in enumerate(tqdm.tqdm(loader)):
                image, label, sample = batch

                image = image.to(configs['device'])
                label = label.to(configs['device'])

                out = model(image)
                predictions = torch.argmax(out,dim=1)

                loss = criterion(input = out, target = label)

                total_loss += loss.item()

                for metric in metrics:
                    if not isinstance(metric, (MulticlassAveragePrecision, MulticlassAUROC)):
                        metric(predictions, label)
                    else:
                        metric(out, label)

                num_samples += image.shape[0]

                del image, label, out, predictions, loss
                gc.collect()

    total_loss = total_loss/num_samples
    log_dict = {'Epoch':epoch, phase + ' loss': total_loss}

    metrics_vals = {}

    for idx, metric in enumerate(metrics):
        if configs['num_classes'] == 1:
            scores = metric.compute()
            metrics_vals[metric.__class__.__name__] = scores
            log_dict[phase + ' ' + metric.__class__.__name__ + ' Class: Deformation'] = scores
        elif metric.average != 'none':
            scores = metric.compute()
            metrics_vals[metric.__class__.__name__] = scores
            log_dict[phase + ' ' + metric.average + ' ' + metric.__class__.__name__] = scores
        else:
            scores = metric.compute()
            metrics_vals[metric.__class__.__name__] = scores
            for idx in range(scores.shape[0]):
                log_dict[phase + ' ' + metric.__class__.__name__ + ' Class: ' + class_dict[idx]] = scores[idx]

    if verbose:
        print(f"GPU Utilization: {torch.cuda.utilization(device=None)}")
        
    if configs['wandb']:
        wandb.log(log_dict)
    elif verbose:
        print(log_dict)

    if epoch == 'Test':
        return metrics_vals
    else:
        return total_loss