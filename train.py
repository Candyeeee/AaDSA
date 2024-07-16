import time
import csv, os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchsummary import summary
from network import GlobalGenerator
from data.read_data import load_image_data, dsa_dataset_withmask, dsa_dataset
from options.train_options import TrainOptions
import train_utils
from visualizer import Visualizer
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # setting parameters
    opt = TrainOptions().parse()
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    use_cuda = torch.cuda.is_available()


    train_data = dsa_dataset_withmask(data_root='/data_new/Dora/DATA/DSA/', mode='train', norm_range_min=0.0,
                 norm_range_max=4095.0)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)

    test_data = dsa_dataset(data_root='/data_new/Dora/DATA/DSA/', mode='test', norm_range_min=0.0,
                 norm_range_max=4095.0)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=False)

    print("Train data size: %d" % len(train_loader))
    print("Test data size: %d" % len(test_loader))

    torch.manual_seed(opt.seed + opt.start_epoch)

    model = GlobalGenerator(1, 1, ngf=32, n_blocks=3, upsample_type='nearest', skip_connection=True)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        # model = model.cuda()
        cudnn.benchmark = True
        print('Using CUDA cudnn...')
        print('Using', torch.cuda.device_count(), 'GPUs.')


    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, betas=(0.9, opt.beta1),eps=1e-8, weight_decay=opt.weight_decay)
    print('optimizer: AdamW...')

    criterion = nn.L1Loss(reduction='mean').cuda()

    log_name = opt.ckpt_dir + opt.name + '_' + str(opt.seed) + '.csv'
    weight_path = opt.ckpt_dir + opt.name + '/'
    visualizer = Visualizer(opt)

    for epoch in range(opt.start_epoch, opt.epoch_num):
        since = time.time()

        ### Train ###
        trn_loss = train_utils.train_withmask(model, train_loader, optimizer, criterion, epoch, opt.epoch_num)

        print('Epoch {:d}:  Train - Loss: {:.4f}'.format(epoch, trn_loss))
        time_elapsed = time.time() - since
        print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Test
        val_loss, current_predresults = train_utils.test(model, test_loader, criterion)

        print('Val - Loss: {:.4f} '.format(val_loss))
        time_elapsed = time.time() - since
        print('Total Time {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


        # checkpoint
        if epoch % opt.save_epoch_freq == 0 or epoch == last_epoch:
            train_utils.save_weights(model, epoch, val_loss, weight_path)






