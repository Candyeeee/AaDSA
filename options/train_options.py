from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='resume a model from check points')
        self.parser.add_argument('--seed', type=int, default=36158957, help='rng seed')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--start_epoch', type=int, default=0, help='the starting epoch')
        self.parser.add_argument('--epoch_num', type=int, default=600, help='maximum training epochs')
        self.parser.add_argument('--mode', type=str, default='train', help='train, val, test, etc')
        # data augmentation: mixup
        self.parser.add_argument('--mixup', type=bool, default=True, help='using mixup data')
        self.parser.add_argument('--mixup_alpha', type=float, default=0.2, help='alpha for beta_randomly mixup')
        self.parser.add_argument('--mix_rate', type=float, default=0.667, help='mix rate for mixup data')

        self.parser.add_argument('--optimizer', type=str, default='AdamW', help='SGD, SGDW，Adam, AdamW, RMSprop')
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for agd or adam')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum term of sgd')
        self.parser.add_argument('--beta1', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: step|plateau')
        self.parser.add_argument('--gamma', type=int, default=0.1,
                                 help='multiply by a gamma every lr_step_size epochs')
        self.parser.add_argument('--lr_step_size', type=int, default=500,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--epoch_count', type=int, default=0,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')


        self.isTrain = True
