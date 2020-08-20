
import torch
from tqdm import tqdm

from pprint import PrettyPrinter
pp = PrettyPrinter()

from utils.utils import AverageMeter, calculate_mAP, detect_objects


def calculate_accuracy(model, data_loader, device, data_type):
    model.eval()  # eval mode disables dropout

    accus = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            images = images.to(dtype=data_type)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            running_corrects = torch.sum(preds == labels.data)
            acc = running_corrects.double().item() / len(labels)
            # print("acc", acc)

            accus.update(acc, len(labels))
            # print("accus.avg", accus.avg)

    return accus.avg



from models.mobilenet_ssd.mobilenet_ssd_priors import priors
priors_cxcy = priors


def eval_objectdetection(model, data_loader, n_classes, label_map, rev_label_map, device):
    model.eval()

    priors_cxcy.to(device)

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(data_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # VGG
            # det_boxes_batch, det_labels_batch, det_scores_batch = models.detect_objects(predicted_locs, predicted_scores,
            #                                                                            min_score=0.01, max_overlap=0.45,
            #                                                                            top_k=200)

            # MobilenetV2
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(
                model,
                priors_cxcy,
                predicted_locs,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.5,
                top_k=200,
                n_classes=n_classes,
                device=device
            )

            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(
            det_boxes,
            det_labels,
            det_scores,
            true_boxes,
            true_labels,
            true_difficulties,
            label_map,
            rev_label_map,
            device
        )

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)
    return mAP


from torchvision import transforms
from PIL import ImageDraw, ImageFont

torch.set_printoptions(profile="30")

def detect(model, original_image, min_score, max_overlap, top_k, device, n_classes, rev_label_map, label_color_map, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # img_data = original_image.load()
    # for i in range(10):
    #     for j in range(10):
    #         print("img x y", j,i, img_data[j, i], end=" ")
    #     print(" ")

    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])


    # origin_tensor = to_tensor(resize(original_image))
    # print("origin_tensor",  torch.flatten(origin_tensor))

    image = normalize(to_tensor(resize(original_image)))

    # print("image flatten", torch.flatten(image))
    # print("image", image.shape)
    # print("image", image)

    # Move to default device
    image = image.to(device)

    # Forward prop.

    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    res = torch.cat([predicted_locs.squeeze(), predicted_scores.squeeze()], dim=1)
    # print("predected_result.shape", res.shape)
    # print("predected_result", res[:5])


    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = detect_objects(
        model,
        priors_cxcy,
        predicted_locs,
        predicted_scores,
        min_score=min_score,
        max_overlap=max_overlap,
        top_k=top_k,
        n_classes=n_classes,
        device=device
    )

    # print("det_boxes", det_boxes)
    # print("det_labels", det_labels)
    # print("det_scores", det_scores)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)

    # print("original_dims", original_dims)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image, det_boxes, det_labels, det_scores

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    #font = ImageFont.truetype("./calibril.ttf", 15)
    font = ImageFont.truetype("data/arial.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image, det_boxes, det_labels, det_scores