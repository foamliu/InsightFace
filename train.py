import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn

from data_gen import ArcFaceDataset
from models import ArcFaceEncoder, ArcMarginModel
from utils import *


def main():
    global best_loss, epochs_since_improvement, checkpoint, start_epoch
    best_loss = 100000
    writer = SummaryWriter()

    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = ArcFaceEncoder()
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=lr)
        model = ArcMarginModel()
        model_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        model = checkpoint['model']
        model_optimizer = checkpoint['model_optimizer']

    # Move to GPU, if available
    encoder = encoder.to(device)
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
            adjust_learning_rate(encoder_optimizer, 0.1)

        # One epoch's training
        train_loss, train_top5_accs = train(train_loader=train_loader,
                                            encoder=encoder,
                                            model=model,
                                            criterion=criterion,
                                            encoder_optimizer=encoder_optimizer,
                                            model_optimizer=model_optimizer,
                                            epoch=epoch)
        train_dataset.shuffle()
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Top5 Accuracy', train_top5_accs, epoch)

        # One epoch's validation
        valid_loss, valid_top5_accs = validate(val_loader=val_loader,
                                               encoder=encoder,
                                               model=model,
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
        save_checkpoint(epoch, epochs_since_improvement, encoder, encoder_optimizer, model, model_optimizer, best_loss,
                        is_best)


def train(train_loader, encoder, model, criterion, encoder_optimizer, model_optimizer, epoch):
    encoder.train()  # train mode (dropout and batchnorm is used)
    model.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (inputs, class_id_true) in enumerate(train_loader):
        # Move to GPU, if available
        inputs = inputs.to(device)
        class_id_true = class_id_true.to(device)  # [N, 1]

        # Forward prop.
        embedding = encoder(inputs)  # embedding => [N, 512]
        class_id_out = model(embedding)  # class_id_out => [N, 10575]

        # Calculate loss
        loss = criterion(class_id_out, class_id_true)

        # Back prop.
        encoder_optimizer.zero_grad()
        model_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(encoder_optimizer, grad_clip)
        clip_gradient(model_optimizer, grad_clip)

        # Update weights
        encoder_optimizer.step()
        model_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(class_id_out, class_id_true, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                   loss=losses,
                                                                                   top5_accs=top5_accs))

    return losses.avg, top5_accs.avg


def validate(val_loader, encoder, model, criterion):
    encoder.eval()  # eval mode (no dropout or batchnorm)
    model.eval()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    with torch.no_grad():
        # Batches
        for i, (inputs, class_id_true) in enumerate(val_loader):
            # Move to GPU, if available
            inputs = inputs.to(device)
            class_id_true = class_id_true.to(device)

            # Forward prop.
            embedding = encoder(inputs)  # embedding => [N, 512]
            class_id_out = model(embedding)  # class_id_out => [N, 10575]

            # Calculate loss
            loss = criterion(class_id_out, class_id_true)

            # Keep track of metrics
            losses.update(loss.item())
            top5_accuracy = accuracy(class_id_out, class_id_true, 5)
            top5_accs.update(top5_accuracy)

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(i, len(val_loader),
                                                                                       loss=losses,
                                                                                       top5_accs=top5_accs))

    return losses.avg


if __name__ == '__main__':
    main()
