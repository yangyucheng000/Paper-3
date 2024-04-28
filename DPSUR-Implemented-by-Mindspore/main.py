from mindspore import context
from model.get_model import get_model
from algorithmDP.DPSUR import DPSUR
import argparse

def run(args):

    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')

    if args.dataset.lower() == 'mnist': #目前实现了对于Mnist数据集的支持
        model = get_model('DPSUR', 'MNIST', 'CPU')
        DPSUR(train_data_name = args.dataset, test_data_name = args.dataset, argModel = model, l2_norm_clip = args.dp_norm, learning_rate= args.learning_rate,
            batch_size = args.batch_size, epsilon_budget=args.dp_epsilon, delta = args.dp_delta, sigma = args.dp_sigma, device = 'cpu', size_valid = args.size_valid,
            use_scattering = args.use_scattering, input_norm = args.input_norm, bn_noise_multiplier = args.bn_noise_multiplier, C_v = args.C_v,
            beta = args.beta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm', "--algorithm", type=str, default='DPSUR')
    parser.add_argument('-data', "--dataset", type=str, default='MNIST')
    parser.add_argument('-lr', "--learning_rate", type=float, default = 0.5)
    parser.add_argument('-dpn', "--dp_norm", type=float, default = 0.1)
    parser.add_argument('-dps', "--dp_sigma", type=float, default = 1.23)
    parser.add_argument('-eps', "--dp_epsilon", type=float, default=1.0)
    parser.add_argument('-delta', "--dp_delta", type=float, default = 1e-5)
    parser.add_argument('-bs', "--batch_size", type = int, default = 256)
    parser.add_argument('--size_valid', type=int, default=5000)
    parser.add_argument('--use_scattering', default= True)
    parser.add_argument("--input_norm", default="BN")
    parser.add_argument('--bn_noise_multiplier', type=float, default=8)
    parser.add_argument('--C_v', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=-1.0)
    args = parser.parse_args()
    run(args)
