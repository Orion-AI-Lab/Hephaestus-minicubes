import pyjson5 as json
from utilities.utils import prepare_configuration
from training import train_supervised 
import argparse
import warnings 
import wandb
import os


if __name__ == "__main__":
    #torch.manual_seed(configs['seed'])
    #torch.cuda.manual_seed(configs['seed'])
    #torch.backends.cudnn.deterministic = True
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    warnings.filterwarnings("ignore", category=UserWarning, module='albumentations')
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=None, required=False)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=None, required=False)

    args = parser.parse_args()

    config_path = "configs/configs.json"
    configs = json.load(open(config_path,'r'))
    if configs['augment']:
        augm = json.load(open('configs/augmentations/augmentation.json','r'))
        if args.image_size is not None:
            augm["augmentations"]["RandomResizedCrop"]["value"] = args.image_size
        configs.update(augm)

    configs['wandb'] = args.wandb

    try:
        train_supervised.train(configs, args.verbose)
    except KeyboardInterrupt:
        print('Training interrupted. Exiting gracefully...')
        if configs['wandb']:
            wandb.finish(exit_code=1)

        os._exit(0)