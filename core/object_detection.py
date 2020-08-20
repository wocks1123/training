import torch
from PIL import Image

import os
import time
import json

from utils.config import load_config
from utils.utils import get_tensortype, AverageMeter, save_checkpoint
from helper.dataloader_helper import load_dataset
from helper.evaluate_helper import calculate_accuracy, eval_objectdetection, detect
from helper.model_helper import load_model
from helper.train_helper import train_objectdetection, validate_objectdetection


from models.mobilenet_ssd.loss import MultiBoxLoss


def training(config_path, transforms=None):
    from utils.logger import logger, log_config, start_time
    # object deteciton pascalvoc07

    config = load_config(config_path)
    log_config(config)

    device = config["device"]
    data_type = get_tensortype(config["data_type"])

    training_option = config["training_option"]
    learning_rate = training_option["learning_rate"]
    momentum = training_option["momentum"]
    weight_decay = training_option["weight_decay"]
    batch_size = training_option["batch_size"]
    n_epochs = training_option["n_epochs"]

    ##########################################################################################################
    # data loader
    dataset_option = config["dataset_option"]
    train_loader, class_list = load_dataset(
        dataset_name=dataset_option["name"],
        data_dir=dataset_option["train_data_dir"],
        split="TRAIN",
        options=dataset_option,
        transforms=transforms["TRAIN"],
        batch_size=batch_size
    )
    val_loader, _ = load_dataset(
        dataset_name=dataset_option["name"],
        data_dir=dataset_option["valid_data_dir"],
        options=dataset_option,
        split="TEST",
        transforms=transforms["TEST"],
        batch_size=batch_size
    )

    ##########################################################################################################
    # set training parameters from config

    model = load_model(
        model_option=config["model_option"],
        num_classes=len(class_list)
    )

    biases = list()
    not_biases = list()
    param_names_biases = list()
    param_names_not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
                param_names_biases.append(param_name)
            else:
                not_biases.append(param)
                param_names_not_biases.append(param_name)
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * learning_rate}, {'params': not_biases}],
                                lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    model = model.to(device)
    logger.info("model : ".format(model))
    criterion = MultiBoxLoss(priors_cxcy=model.priors).to(device)

    # lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    if training_option["use_scheduler"] is True:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_option["lr_stepsize"],
            gamma=training_option["lr_gamma"]
        )
    else:
        lr_scheduler = None

    logger.info("===============      model      ===============")
    logger.info(model)
    logger.info("===============    optimizer    ===============")
    logger.info(optimizer)

    model.to(dtype=data_type)
    model = model.to(device)
    criterion = criterion.to(device)
    ##########################################################################################################
    dest_folder = f"data/weight/{start_time}_{config['model_option']['model']}_{dataset_option['name']}"  # directory for saving model.pt
    if os.path.exists(dest_folder) is False:
        os.mkdir(dest_folder)

    epochs_since_improvement = 0
    best_loss = 100.

    epoch_time = AverageMeter()
    loss_avg = AverageMeter()

    start = time.time()
    for epoch in range(n_epochs):
        train_loss = train_objectdetection(
            data_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            # scheduler=lr_scheduler,
            num_epochs=epoch,
            device=device,
            # data_type=data_type
        )

        loss_avg.update(train_loss, 1)
        logger.info(f"train loss {loss_avg.val} ({loss_avg.avg})")

        logger.info(f"Learning Rate : {optimizer.param_groups[0]['lr']}")

        if lr_scheduler is not None:
            lr_scheduler.step()

        val_loss = validate_objectdetection(
            data_loader=val_loader,
            model=model,
            criterion=criterion,
            device=device
            # data_type=data_type
        )

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        epoch_time.update(time.time() - start)
        start = time.time()
        logger.info("epoch time {epoch_time.val} ({epoch_time.avg:.3f})".format(epoch_time=epoch_time))

        save_checkpoint(
            epoch,
            epochs_since_improvement,
            model,
            optimizer,
            val_loss,
            best_loss,
            is_best,
            dest_folder
        )


def evaluation(config_path, checkpoint_path, transforms=None):
    config = load_config(config_path)

    device = config["device"]
    dataset_option = config["dataset_option"]

    val_loader, class_list = load_dataset(
        dataset_name=dataset_option["name"],
        data_dir=dataset_option["test_data_dir"],
        split="TEST",
        options=dataset_option,
        transforms=transforms["TEST"],
        batch_size=32
    )

    # checkpoint = "data/weight/2020-08-14_12_20_44_mobilenetv2_q_ssd_pascalvoc_v2/epoch390_checkpoint.pth.tar"  # MobilenetV2 SSD
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = checkpoint['model']
    print("model", model)

    label_map = {k: v + 1 for v, k in enumerate(class_list)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

    ret = eval_objectdetection(
        model=model,
        data_loader=val_loader,
        n_classes=len(class_list) + 1,
        label_map=label_map,
        rev_label_map=rev_label_map,
        device=device
    )
    print(ret)



def inference(config_path, checkpoint_path, img_path, transforms=None):
    config = load_config(config_path)

    device = config["device"]
    data_type = get_tensortype(config["data_type"])

    dataset_option = config["dataset_option"]

    _, class_list = load_dataset(
        dataset_name=dataset_option["name"],
        data_dir=dataset_option["test_data_dir"],
        split="TEST",
        options=dataset_option,
        transforms=transforms["TEST"],
        batch_size=32
    )
    print("class_list", class_list)
    # checkpoint = "data/weight/2020-08-06_14_24_54_vgg16/epoch10_checkpoint.pth.tar"  # vgg

    checkpoint = "data/weight/BEST_checkpoint_ssd300_mobilenetv2_pascal.pth.tar"
    checkpoint = "data/weight/2020-08-06_15_30_46_mobilenetv2/epoch20_checkpoint.pth.tar"  # mobilenet
    # checkpoint = "data/weight/2020-08-07_21_34_22_mobilenetv2_ssd/BEST_checkpoint.pth.tar"  # MobilenetV2 SSD
    checkpoint = "data/weight/2020-08-07_21_34_22_mobilenetv2_ssd/epoch260_checkpoint.pth.tar"  # MobilenetV2 SSD
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = checkpoint['model']
    # print("model", model)

    label_map = {k: v + 1 for v, k in enumerate(class_list)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                       '#808000',
                       '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

    # img_path = "/DB/nsfw/kait/batch_yesfile_24738402_1.jpg"
    img_path = "/DB/black.jpg"

    img_path = "000005.jpg"
    img_path = "/DB/CIFAR-10/test500/airplane/0001.jpg"
    img_path = "6line.jpg"
    img_path = "black_bottle.jpg"
    img_path = "/DB/VOC2007_test100/JPEGImages/000005.jpg"
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    res, det_boxes, det_labels, det_scores = detect(
        model,
        original_image,
        min_score=0.2,
        max_overlap=0.5,
        top_k=200,
        device=device,
        n_classes=len(class_list) + 1,
        rev_label_map=rev_label_map,
        label_color_map=label_color_map
    )
    print("det_boxes", det_boxes)
    print("det_labels", det_labels)
    print("det_scores", det_scores)
    res.save("result.jpg")


def eval_test100(config_path, checkpoint_path):
    """
    for pascal voc
    """

    config = load_config(config_path)

    dataset_option = config["dataset_option"]

    _, class_list = load_dataset(
        dataset_name=dataset_option["name"],
        # data_dir=dataset_option["test_data_dir"],
        data_dir="data/json/VOCTEST1000",
        split="TEST",
        options=dataset_option,
        transforms=None,
        batch_size=1
    )

    # checkpoint = "data/weight/2020-08-07_21_34_22_mobilenetv2_ssd/epoch260_checkpoint.pth.tar"  # MobilenetV2 SSD
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model = checkpoint['model']
    # print("model", model)

    label_map = {k: v + 1 for v, k in enumerate(class_list)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                       '#808000',
                       '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

    with open('data/json/VOCTEST1000_images.json', 'r') as j:
        images = json.load(j)
    with open('data/json/VOCTEST1000_objects.json', 'r') as j:
        objects = json.load(j)
    print("images", images)
    for i in range(len(images)):
        print("images[i]", images[i])
        original_image = Image.open(images[i], mode='r')
        original_image = original_image.convert('RGB')

        curr_ob = objects[i]

        fp = open(f"/root/calmap/mAP/input/ground-truth/{i}.txt", "w")
        for j in range(len(curr_ob["boxes"])):
            gt_boxes = curr_ob["boxes"][j]
            gt_labels = curr_ob["labels"][j] - 1
            gt_difficulties = curr_ob["difficulties"][j]
            line = f"{class_list[gt_labels]} {gt_boxes[0]} {gt_boxes[1]} {gt_boxes[2]} {gt_boxes[3]}"
            fp.write(line + "\n")

        fp.close()

        res, det_boxes, det_labels, det_scores = detect(
            model,
            original_image,
            min_score=0.2,
            max_overlap=0.5,
            top_k=200,
            device="cuda",
            n_classes=len(class_list) + 1,
            rev_label_map=rev_label_map,
            label_color_map=label_color_map
        )

        fp = open(f"/root/calmap/mAP/input/desktop_detection-results/{i}.txt", "w")
        for j in range(len(det_boxes)):
            dt_box = det_boxes[j]
            dt_score = det_scores[0][j]
            dt_label = det_labels[j]
            line = f"{dt_label} {dt_score} {int(dt_box[0])} {int(dt_box[1])} {int(dt_box[2])} {int(dt_box[3])}"
            fp.write(line + "\n")
        fp.close()