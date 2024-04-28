import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--parameter',  default='./GIE.ckpt', help='name of parameter file')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument("--device_id", type=str, default='1', help="Device id")
parser.add_argument('--device_target', type=str, default="GPU",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target')

parser.add_argument('--rgb_root', type=str, default='Dataset/VT5000/VT5000_clear/Train/RGB/', help='the training rgb images root')
parser.add_argument('--t_root', type=str, default='Dataset/VT5000/VT5000_clear/Train/T/', help='the training t images root')
parser.add_argument('--gt_root', type=str, default='Dataset/VT5000/VT5000_clear/Train/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='Dataset/VT5000/VT5000_clear/Test/RGB/', help='the test rgb images root')
parser.add_argument('--test_t_root', type=str, default='Dataset/VT5000/VT5000_clear/Test/T/', help='the test t images root')
parser.add_argument('--test_gt_root', type=str, default='Dataset/VT5000/VT5000_clear/Test/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='TNet/', help='the path to save models and logs')
opt = parser.parse_args()