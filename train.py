import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn

from data_gen import ArcFaceDataset
from models import ArcFaceModel
from utils import *


def main():
    global best_loss, epochs_since_improvement, checkpoint, start_epoch, l1_criterion
    best_loss = 100000
    writer = SummaryWriter()

    # Initialize / load checkpoint
    if checkpoint is None:
        model = ArcFaceModel()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                     weight_decay=5e-4)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)
    val_dataset = ArcFaceDataset('valid')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                             pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.1)

        # One epoch's training
        train_loss, train_gen_accs, train_age_mae = train(train_loader=train_loader,
                                                          model=model,
                                                          criterion=criterion,
                                                          optimizer=optimizer,
                                                          epoch=epoch)
        train_dataset.shuffle()
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Gender Accuracy', train_gen_accs, epoch)
        writer.add_scalar('Train Age MAE', train_age_mae, epoch)

        # One epoch's validation
        valid_loss, valid_gen_accs, valid_age_mae = validate(val_loader=val_loader,
                                                             model=model,
                                                             criterion=criterion)

        writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('Valid Gender Accuracy', valid_gen_accs, epoch)
        writer.add_scalar('Valid Age MAE', valid_age_mae, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (inputs, class_id_true) in enumerate(train_loader):
        chunk_size = inputs.size()[0]
        # Move to GPU, if available
        inputs = inputs.to(device)
        class_id_true = class_id_true.to(device)  # [N, 1]

        # Forward prop.
        class_id_out = model(inputs)  # age_out => [N, 1], gen_out => [N, 2]

        # Calculate loss
        loss = criterion(class_id_out, class_id_true)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), chunk_size)
        top5_accuracy = accuracy(class_id_out, class_id_true, 5)
        top5_accs.update(top5_accuracy, chunk_size)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                   loss=losses,
                                                                                   top5_accs=top5_accs))

    return losses.avg


def validate(val_loader, model, criterion):
    model.eval()  # eval mode (no dropout or batchnorm)

    losses = AverageMeter()
    top5_accs = AverageMeter()

    with torch.no_grad():
        # Batches
        for i, (inputs, class_id_true) in enumerate(val_loader):
            chunk_size = inputs.size()[0]
            # Move to GPU, if available
            inputs = inputs.to(device)
            class_id_true = class_id_true.to(device)

            # Forward prop.
            class_id_out = model(inputs)

            # Calculate loss
            loss = criterion(class_id_out, class_id_true)

            # Keep track of metrics
            losses.update(loss.item(), chunk_size)
            top5_accuracy = accuracy(class_id_out, class_id_true, 5)
            top5_accs.update(top5_accuracy, chunk_size)

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(i, len(val_loader),
                                                                                       loss=losses,
                                                                                       top5_accs=top5_accs))

    return losses.avg


if __name__ == '__main__':
    main()
