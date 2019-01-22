import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn

from data_gen import ArcFaceDataset
from models import ArcFaceEncoder, ArcMarginModel
from utils import *


def main():
    global best_loss, epochs_since_improvement, checkpoint, start_epoch, train_steps
    best_loss = 100000
    writer = SummaryWriter()
    train_steps = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = ArcFaceEncoder()
        model = nn.DataParallel(model)
        metric_fc = ArcMarginModel()
        metric_fc = nn.DataParallel(metric_fc)

        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=lr,
                                     weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        model = nn.DataParallel(model)
        metric_fc = checkpoint['metric_fc']
        metric_fc = nn.DataParallel(metric_fc)
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)
    val_dataset = ArcFaceDataset('valid')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                             pin_memory=True)

    reduced_20k = reduced_28k = False

    # Epochs
    for epoch in range(start_epoch, epochs):
        if train_steps >= 20 * 1024 and not reduced_20k:
            adjust_learning_rate(optimizer, 0.1)
            print('reduced lr at 20k')
            reduced_20k = True
        if train_steps >= 28 * 1024 and not reduced_28k:
            adjust_learning_rate(optimizer, 0.1)
            print('reduced lr at 28k')
            reduced_28k = True

        # One epoch's training
        train_loss, train_top5_accs = train(train_loader=train_loader,
                                            model=model,
                                            metric_fc=metric_fc,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            epoch=epoch)
        train_dataset.shuffle()
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Top5 Accuracy', train_top5_accs, epoch)

        # One epoch's validation
        valid_loss, valid_top5_accs = validate(val_loader=val_loader,
                                               model=model,
                                               metric_fc=metric_fc,
                                               criterion=criterion)

        writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('Valid Top5 Accuracy', valid_top5_accs, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, best_loss, is_best)


def train(train_loader, model, metric_fc, criterion, optimizer, epoch):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)
        label = label.to(device)  # [N, 1]
        # print('class_id_true.size(): ' + str(class_id_true.size()))

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]
        # print('embedding.size(): ' + str(embedding.size()))
        pred = metric_fc(feature, label)  # class_id_out => [N, 10575]
        # print('class_id_out.size(): ' + str(class_id_out.size()))

        # Calculate loss
        loss = criterion(pred, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(pred, label, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                   loss=losses,
                                                                                   top5_accs=top5_accs))
        global train_steps
        train_steps += 1

    return losses.avg, top5_accs.avg


def validate(val_loader, model, metric_fc, criterion):
    model.eval()  # eval mode (no dropout or batchnorm)
    metric_fc.eval()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    with torch.no_grad():
        # Batches
        for i, (img, label) in enumerate(val_loader):
            # Move to GPU, if available
            img = img.to(device)
            label = label.to(device)

            # Forward prop.
            feature = model(img)  # embedding => [N, 512]
            pred = metric_fc(feature, label)  # class_id_out => [N, 10575]

            # Calculate loss
            loss = criterion(pred, label)

            # Keep track of metrics
            losses.update(loss.item())
            top5_accuracy = accuracy(pred, label, 5)
            top5_accs.update(top5_accuracy)

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(i, len(val_loader),
                                                                                       loss=losses,
                                                                                       top5_accs=top5_accs))

    return losses.avg, top5_accs.avg


if __name__ == '__main__':
    main()
