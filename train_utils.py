import numpy as np
from torch.autograd import Variable
import torch
import os
import random
import torch.nn.functional as F
from collections import OrderedDict
    


def train(model, trn_loader, optimizer, criterion, epoch, num_epochs):
    scale_weight = 1.
    model.train()
    trn_loss = 0
    dataset_size = len(trn_loader)

    tv_loss = TVLoss(1.).cuda()

    epoch_iter = 0
    for idx, (inputs, dsa) in enumerate(trn_loader):

        if (dataset_size*epoch + epoch_iter) % 4 == 0:
            current_lr, current_mom = onecycle_schedule(dataset_size*epoch + epoch_iter, num_epochs, dataset_size,
                                                                    max_lr=0.01, div=10, pct=90, max_mom=0.9, min_mom=0.8)

            adjust_learning_rate(optimizer, current_lr, current_mom)

        epoch_iter += 1

        inputs = inputs.cuda()
        dsa = dsa.cuda()

        # start back propagation
        optimizer.zero_grad()
        output0 = model(inputs)
        

        loss = criterion(output0, dsa)
        loss.backward()
        optimizer.step()
        # end one back propagation loop

        trn_loss += loss.item()

        if epoch_iter % 10 == 0:
            print('Epoch {:d}, iter {:d}  Train - Loss: {:.4f}'.format(
                epoch+1, epoch_iter, trn_loss / epoch_iter))

    print('current learning_rate: {:.6f} momentum: {:.6f}'.format(
        optimizer.param_groups[0]['lr'], 0))
    trn_loss /= dataset_size

    return trn_loss


def train_withmask(model, trn_loader, optimizer, criterion, epoch, num_epochs):
    scale_weight = 1.
    model.train()
    trn_loss = 0
    dataset_size = len(trn_loader)

    epoch_iter = 0
    for idx, (inputs, dsa, mask) in enumerate(trn_loader):

        if (dataset_size*epoch + epoch_iter) % 4 == 0:
            current_lr, current_mom = onecycle_schedule(dataset_size*epoch + epoch_iter, num_epochs, dataset_size,
                                                                    max_lr=0.01, div=10, pct=90, max_mom=0.9, min_mom=0.8)

            adjust_learning_rate(optimizer, current_lr, current_mom)

        epoch_iter += 1

        inputs = inputs.cuda()
        dsa = dsa.cuda()
        mask = mask.cuda()

        # start back propagation
        optimizer.zero_grad()

        output0 = model(inputs)

        loss = criterion(output0*mask, dsa*mask)

        loss.backward()
        optimizer.step()
        # end one back propagation loop

        trn_loss += loss.item()

        if epoch_iter % 10 == 0:
            print('Epoch {:d}, iter {:d}  Train - Loss: {:.4f}'.format(
                epoch+1, epoch_iter, trn_loss / epoch_iter))

    print('current learning_rate: {:.6f} momentum: {:.6f}'.format(
        optimizer.param_groups[0]['lr'], 0))
    trn_loss /= dataset_size

    return trn_loss




def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.
    current_predresults = OrderedDict()
    dataset_size = len(test_loader)
    epoch_iter = 0
    for inputs, dsa in test_loader:
        epoch_iter += 1
        inputs = inputs.cuda()
        dsa = dsa.cuda()

        with torch.no_grad():
            output0 = model(inputs)  # output1

        loss = criterion(output0, dsa)

        test_loss += loss.item()
        output0 = output0.detach()

        if epoch_iter == dataset_size:
            current_predresults['inputs'] = torch.clamp(inputs.cpu(), 0., 1.)
            current_predresults['dsa'] = torch.clamp(dsa.cpu(), 0., 1.)
            current_predresults['pred_dsa'] = torch.clamp(output0.cpu(), 0., 1.)

    test_loss /= dataset_size
    return test_loss, current_predresults


def adjust_learning_rate(optimizer, lr, mom=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def onecycle_schedule(current_iter, epochs, epoch_iters, max_lr, div=10, pct=95, max_mom=0.95, min_mom=0.85):
    half_cyc = 0.5 * pct / 100. * epochs
    half_iters = half_cyc * epoch_iters
    min_lr = max_lr / div
    if current_iter <= half_iters:
        lr = min_lr + current_iter / half_iters * (max_lr - min_lr)
        mom = max_mom - current_iter / half_iters * (max_mom - min_mom)
    else:
        if current_iter <= 2 * half_iters:
            lr = max_lr - (current_iter - half_iters) / half_iters * (max_lr - min_lr)
            mom = min_mom + (current_iter - half_iters) / half_iters * (max_mom - min_mom)
        else:
            lr = min_lr - (current_iter - 2*half_iters) / (epochs * epoch_iters - 2*half_iters) * (min_lr - min_lr / div)
            mom = max_mom
    return lr, mom


def calculate_gradient2(img):
    kernel = build_diff_kernel()
    img = F.pad(img, [1, 1, 1, 1], mode='replicate')
    return F.conv2d(img, kernel, padding=0)


def build_diff_kernel(cuda=True):
    kernel = np.zeros([2, 1, 3, 3], dtype=np.float32)
    kernel[0, 0, 1, 0] = 1.
    kernel[0, 0, 1, 1] = -1.
    kernel[1, 0, 0, 1] = 1.
    kernel[1, 0, 1, 1] = -1.
    kernel = torch.FloatTensor(kernel)

    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def save_weights(model, epoch, loss, weight_path):
    weights_fname = 'weights-%d-%.3f.pth' % (epoch, loss)
    weights_fpath = os.path.join(weight_path, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss': loss,
            'state_dict': model.state_dict()
        }, weights_fpath)


def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {})"
          .format(startEpoch-1, weights['loss']))
    return startEpoch



