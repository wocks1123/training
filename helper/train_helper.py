import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import time

from utils.utils import *
from utils.logger import logger


def train_classification(data_loader, model, criterion, optimizer, scheduler, num_epochs, device, data_type):
    """
    - training for image classification
    - One epoch's training.

    :param data_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param device:
    :return:
    """
    print_freq = 200  # print training or validation status every __ batches
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, labels) in enumerate(data_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(dtype=data_type)
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        labels = labels.to(device)

        # Forward prop.
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # # Clip gradients, if necessary
        # if grad_clip is not None:
        #     clip_gradient(optimizer, grad_clip)

        # Update models
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    num_epochs,
                    i,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses
                )
            )


def validate_classification(data_loader, model, criterion, device, data_type):
    """
    - validation for image classification
    - One epoch's validation

    :param data_loader:
    :param model:
    :param criterion:
    :param device:
    :return: average loss
    """
    print_freq = 200
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, labels) in enumerate(data_loader):
            # Move to default device
            images = images.to(device)
            labels = labels.to(device)

            images = images.to(dtype=data_type)


            # Forward prop.
            outputs = model(images)

            # Loss
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                logger.info(
                    '[{0}/{1}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        i,
                        len(data_loader),
                        batch_time=batch_time,
                        loss=losses
                    )
                )

    logger.info('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg



def train_objectdetection(data_loader, model, criterion, optimizer, num_epochs, device):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    print_freq = 200  # print training or validation status every __ batches
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    # Batches
    for i, (images, boxes, labels, _) in enumerate(data_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss

        # for i in range(len(boxes)):
        #  boxes[i] = boxes[i].to('cpu')
        #  labels[i] = labels[i].to('cpu')
        # print (predicted_locs, predicted_scores)
        # print (predicted_locs.shape, predicted_scores.shape)
        # print (len(boxes), len(labels))
        # print (boxes[1], labels[1])
        predicted_locs = predicted_locs.to(device)
        predicted_scores = predicted_scores.to(device)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # # Clip gradients, if necessary
        # if grad_clip is not None:
        #     clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(num_epochs, i, len(data_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

    return losses.avg


def validate_objectdetection(data_loader, model, criterion, device):
    """
    One epoch's validation.
    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :return: average validation loss
    """
    print_freq = 200
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(data_loader):

            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            predicted_locs = predicted_locs.to(device)
            predicted_scores = predicted_scores.to(device)

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                logger.info('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(data_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    logger.info('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg
