import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--arch', type=str, default='DilatedCNN', help='network arch')
        self.parser.add_argument('--crop_size', type=int, default=512, help='crop images to this size')
        self.parser.add_argument('--norm', type=str, default='BatchNormalization',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--horizontal_flip', type=bool, default=False,
                                 help='flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='xavier',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.parser.add_argument('--name', type=str, default='vDSA_wnewdata_wmask', help='name of the experiment')
        self.parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--ckpt_dir', type=str, default='/data_new/Dora/Results/DSA/', help='models are saved here')

        self.parser.add_argument('--no_html', type=bool, default=False, help='display in html')
        self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_ncols', type=int, default=2, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=20,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--display_winsize', type=int, default=1024, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        opt.isTrain = self.isTrain   # train or test

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(opt.ckpt_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        self.opt = opt

        return self.opt

