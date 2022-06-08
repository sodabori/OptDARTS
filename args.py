import argparse
import utils

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser("sota")

        parser.add_argument('--data', type=str, default='../data/CIFAR/', help='location of the data corpus')
        parser.add_argument('--bench-data', type=str, default='../data/NAS-Bench-201/NAS-Bench-201-v1_1-096897.pth', help='location of the benchmark corpus')
        parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
        parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
        parser.add_argument('--exp-path', type=str, default='../exps/', help='path to exp')
        parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
        parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
        parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
        parser.add_argument('--perturb_alpha', type=str, default='none', help='perturb for alpha')
        parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')

        # optimizer settings
        parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer for weight parameters')
        parser.add_argument('--arch_optimizer', type=str, default='adam', help='optimizer for arch parameters')

        # model hyperparameter settings
        # sgd
        parser.add_argument('--sgd_learning_rate', type=float, default=0.025, help='sgd init learning rate')
        parser.add_argument('--sgd_learning_rate_min', type=float, default=0.001, help='sgd min learning rate')
        parser.add_argument('--sgd_momentum', type=float, default=0.9, help='sgd momentum')
        parser.add_argument('--sgd_weight_decay', type=float, default=3e-4, help='sgd weight decay')  # DARTS: 3e-4  RDARTS: 81e-4
        # adam
        parser.add_argument('--adam_learning_rate', type=float, default=1e-3, help='sgd learning rate for')
        parser.add_argument('--adam_learning_rate_min', type=float, default=1e-4, help='sgd learning rate for')
        parser.add_argument('--adam_beta1', type=float, default=0.5, help='sgd beta1 for')
        parser.add_argument('--adam_beta2', type=float, default=0.999, help='sgd beta2 for')
        parser.add_argument('--adam_weight_decay', type=float, default=3e-4, help='sgd weight decay')
        # architecture hyparameter settings
        # sgd
        parser.add_argument('--arch_sgd_learning_rate', type=float, default=0.01, help='adam init learning rate for arch encoding')
        parser.add_argument('--arch_sgd_momentum', type=float, default=0.9, help='adam momentum for arch encoding')
        parser.add_argument('--arch_sgd_weight_decay', type=float, default=3e-4, help='adam weight decay for arch encoding')  # DARTS: 3e-4  RDARTS: 81e-4
        # adam
        parser.add_argument('--arch_adam_learning_rate', type=float, default=1e-3, help='adam learning rate for arch encoding')
        parser.add_argument('--arch_adam_beta1', type=float, default=0.5, help='adam beta1 for arch encoding')
        parser.add_argument('--arch_adam_beta2', type=float, default=0.999, help='adam beta2 for arch encoding')
        parser.add_argument('--arch_adam_weight_decay', type=float, default=3e-4, help='adam weight decay for arch encoding')

        # darts- settings
        parser.add_argument('--auxiliary_skip', action='store_true', default=False, help='use auxiliary skip connection')
        parser.add_argument('--auxiliary_operation', type=str, default='skip', help='operation for auxiliary skip connection')
        parser.add_argument('--decay', default='linear', choices=[None, 'cosine', 'slow_cosine','linear'], help='select scheduler decay on epochs')
        parser.add_argument('--skip_beta', type=float, default=1.0, help='ratio to overshoot or discount auxiliary skip')
        parser.add_argument('--decay_start_epoch', type=int, default=0, help='epoch to start decay')
        parser.add_argument('--decay_stop_epoch', type=int, default=100, help='epoch to stop decay')
        parser.add_argument('--decay_max_epoch', type=int, default=100, help='max epochs to decay')

        # visualization
        parser.add_argument('--exp-name', type=str, default='None', help='target folder for visualization')
        parser.add_argument('--x', type=str, default='-1:1:51', help='A string with format xmin:x_max:xnum') #51
        parser.add_argument('--y', type=str, default='-1:1:51', help='A string with format ymin:y_max:ynum') #51
        parser.add_argument('--show', action='store_true', default=False, help='show graph before saving')
        parser.add_argument('--azim', type=float, default=-60, help='azimuthal angle for 3d landscape')
        parser.add_argument('--elev', type=float, default=30, help='elevation angle for 3d landscape')

        self.args = parser.parse_args()
        print(self.args)

args = Parser().args
beta_decay_scheduler = utils.DecayScheduler(base_lr=args.skip_beta, 
                                            T_max=args.decay_max_epoch, 
                                            T_start=args.decay_start_epoch, 
                                            T_stop=args.decay_stop_epoch, 
                                            decay_type=args.decay)