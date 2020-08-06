
import torchvision.models as models
import torch.nn as nn


def set_parameter_requires_grad(model, feature_extract_flag: bool):
    if feature_extract_flag:
        for param in model.parameters():
            param.requires_grad = False


def load_model(model_name: str, pretrained: bool, num_classes: int, feature_extract_flag: bool=False):
    if model_name == "resnet1818":
        model = models.resnet18(pretrained=pretrained)
        set_parameter_requires_grad(model, feature_extract_flag)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=pretrained)
        set_parameter_requires_grad(model, feature_extract_flag)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    # elif model_name == "mnasNet":
    #     model = models.mnasnet1_0(pretrained=pretrained)
    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=pretrained)
        set_parameter_requires_grad(model, feature_extract_flag)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif model_name == "vgg11_bn":
        model = models.vgg11_bn(pretrained=pretrained)
        set_parameter_requires_grad(model, feature_extract_flag)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        set_parameter_requires_grad(model, feature_extract_flag)
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, num_classes)])  # Add our layer with 4 outputs
        model.classifier = nn.Sequential(*features)  # Replace the model classifier
    # elif model_name == "shufflenet":
    #     model = models.shufflenet_v2_x1_0(pretrained=pretrained)
    elif model_name == "densenet":
        model = models.densenet161(pretrained=pretrained)
        set_parameter_requires_grad(model, feature_extract_flag)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise Exception("Wrong Model Name... Check config.json")

    return model

