
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.quantization as q_models

from models.quantization.MobileNetV2_q import MobileNetV2
from models.mobilenet_ssd.mobilev2ssd import SSD


def load_pretrained_mobilenet(model, model_file):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model


def set_parameter_requires_grad(model, feature_extract_flag: bool):
    if feature_extract_flag:
        for param in model.parameters():
            param.requires_grad = False


def load_model(model_option: dict, num_classes: int):
    model_name = model_option["model"]
    if model_name == "resnet18":
        model = models.resnet18(pretrained=model_option["pretrained"])
        set_parameter_requires_grad(model, model_option["feature_extract_flag"])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=model_option["pretrained"])
        set_parameter_requires_grad(model, model_option["feature_extract_flag"])
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    elif model_name == "mobilenetv2_q":
        model = MobileNetV2(
            num_classes=num_classes,
            width_mult=model_option["width_mult"],
            pretrained=model_option["pretrained"]
        )

        set_parameter_requires_grad(model, model_option["feature_extract_flag"])
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    # elif model_name == "mnasNet":
    #     models = models.mnasnet1_0(pretrained=model_option["pretrained"])
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=model_option["pretrained"])
        set_parameter_requires_grad(model, model_option["feature_extract_flag"])
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif model_name == "vgg11_bn":
        model = models.vgg11_bn(pretrained=model_option["pretrained"])
        set_parameter_requires_grad(model, model_option["feature_extract_flag"])
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=model_option["pretrained"])
        set_parameter_requires_grad(model, model_option["feature_extract_flag"])
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, num_classes)])  # Add our layer with 4 outputs
        model.classifier = nn.Sequential(*features)  # Replace the models classifier
    # elif model_name == "shufflenet":
    #     models = models.shufflenet_v2_x1_0(pretrained=model_option["pretrained"])
    elif model_name == "densenet":
        model = models.densenet161(pretrained=model_option["pretrained"])
        set_parameter_requires_grad(model, model_option["feature_extract_flag"])
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenetv2_q_ssd":
        model = SSD(num_classes=num_classes + 1, backbone_network=model_option["backbone"])
    #############################################################################################################
    # Quantized Models
    elif model_name == "":
        model = q_models.mobilenet_v2()
    else:
        raise Exception("Wrong Model Name... Check config.json " + model_name)

    return model

