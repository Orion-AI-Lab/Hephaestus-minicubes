import timm
import pyjson5 as json
import segmentation_models_pytorch as smp
from models.vit import ViT


def freeze_encoder(model, freeze_percentage):
    encoder_layers = list(model.encoder.children())
    total_layers = len(encoder_layers)
    
    # Calculate the number of layers to freeze based on the percentage
    layers_to_freeze = int(total_layers * freeze_percentage)
    
    # Freeze the specified percentage of layers
    for i, layer in enumerate(encoder_layers):
        if i < layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
    
    print(f"Freezing {layers_to_freeze} layers out of {total_layers} layers in the encoder.")
    return model


def print_model_layers(model):
    for name, param in model.named_parameters():
        layer_trainable = param.requires_grad
        print(f"Layer: {name} | Trainable: {layer_trainable}")

    print("\nDetailed layer structure and their trainable status:")
    for name, module in model.named_children():
        print(f"Module: {name} ({module.__class__.__name__})")
        for layer_name, param in module.named_parameters(recurse=False):
            layer_trainable = param.requires_grad
            print(f"  Layer: {layer_name} | Trainable: {layer_trainable}")


def create_model(configs):
    num_classes = configs['num_classes']

    if configs['task'] == 'segmentation':
        if "unet" in configs["architecture"].lower():
            unet_config = json.load(open('configs/method/unet/unet.json','r'))
            #Overwrite the backbone if specified in the config
            try:
                unet_config['backbone'] = configs['model_backbone']
            except KeyError:
                raise KeyError("No backbone key in the config file")
            
            model = smp.Unet(
                encoder_name=unet_config[
                    "backbone"
                ],
                encoder_weights=unet_config[
                    "encoder_weights"
                ],  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels']))*configs['timeseries_length'],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=num_classes,  # model output channels (number of classes in your dataset)
            )
        elif "deeplabv3" in configs["architecture"].lower():
            deeplabv3_config = json.load(open('configs/method/deeplabv3/deeplabv3.json','r'))
            #Overwrite the backbone if specified in the config
            try:
                deeplabv3_config['backbone'] = configs['model_backbone']
            except KeyError:
                raise KeyError("No backbone key in the config file")
            
            if configs['mask_target'] == 'all':
                classes = configs['timeseries_length']*num_classes
            else:
                classes = num_classes
    
            model = smp.DeepLabV3Plus(
                encoder_name=deeplabv3_config[
                    "backbone"
                ],
                encoder_weights=deeplabv3_config[
                    "encoder_weights"
                ],  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels']))*configs['timeseries_length'],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=classes,  # model output channels (number of classes in your dataset)
            )
            if deeplabv3_config['freeze_encoder'] is not False:
                freeze_encoder(model, deeplabv3_config['freeze_encoder'])
        elif "segformer" in configs["architecture"].lower():
            segformer_config = json.load(open('configs/method/segformer/segformer.json','r'))
            #Overwrite the backbone if specified in the config
            try:
                segformer_config['backbone'] = configs['model_backbone']
            except KeyError:
                raise KeyError("No backbone key in the config file")
            
            model = smp.Segformer(
                encoder_name=segformer_config[
                    "backbone"
                ],
                encoder_weights=segformer_config[
                    "encoder_weights"
                ],
                in_channels=(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels']))*configs['timeseries_length'],
                classes=num_classes,
            )
        else:
            print(f"Model not explicitly included. Trying to load {configs['architecture']} from timm")
            model = timm.models.create_model(
                configs["architecture"].lower(), pretrained=True, num_classes=num_classes, in_chans=(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels']))*configs['timeseries_length']
            )
    elif configs['task'] == 'classification':
        if 'vit' in configs['architecture'].lower():
            model = ViT(num_classes, configs)
        elif 'efficientnet' in configs["architecture"].lower():
            model = timm.create_model(configs["architecture"].lower(), pretrained=False, num_classes=num_classes, in_chans=(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels']))*configs['timeseries_length'])
        else:
            print(f"Model not explicitly included. Trying to load {configs['architecture']} from timm")
            model = timm.create_model(
                configs["architecture"].lower(), pretrained=True, num_classes=num_classes, in_chans=(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels']))*configs['timeseries_length']
            )
    else:
        raise NotImplementedError(f"Task {configs['task']} not implemented for classification task")
    
    return model
