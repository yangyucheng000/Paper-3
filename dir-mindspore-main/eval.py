import argparse
import os

import mindspore
from mindspore import nn, context, ops, Tensor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset import GeneratorDataset

from main import get_args_parser

from models import build_model
from datasets import build_dataset, preprocess_fn, HICOEvaluator


def run_eval(args):

    context.set_context(mode=args.ms_mode, device_target=args.device_target, max_call_depth=2000)
    if args.device_target == "Ascend":
        device_id = int(os.getenv("DEVICE_ID", 0))
        context.set_context(device_id=device_id)
    elif args.device_target == "GPU" and args.ms_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    # Set Parallel
    if args.is_parallel:
        init()
        args.rank, args.rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(device_num=args.rank_size, parallel_mode=parallel_mode, gradients_mean=True, parameter_broadcast=True)
    else:
        args.rank, args.rank_size = 0, 1

    print(f"rank_id: {args.rank}, rank_size: {args.rank_size}")

    img_set = 'val'
    dataset_val = build_dataset(image_set=img_set, args=args)
    
    data_loader_val = GeneratorDataset(source=dataset_val, column_names=["img", "target"])

    compose_map_fn = (lambda img, target: preprocess_fn(args, img_set, img, target))

    data_loader_val = data_loader_val.map(input_columns=["img", "target"], output_columns=["tensor", "mask", "target"], 
                                          num_parallel_workers=8, operations=compose_map_fn)
    data_loader_val = data_loader_val.project(["tensor", "mask", "target"])

    model, postprocessors = build_model(args)
    model.set_train(False)

    # load checkpoint
    param_dict = load_checkpoint(args.resume)
    load_param_into_net(model, param_dict)

    results = []
    targets = []

    for i, (img, mask, target) in enumerate(data_loader_val.create_dict_iterator(output_numpy=True)):
        img = Tensor(img, mindspore.float32)
        mask = Tensor(mask, mindspore.float32)
        output = model(img, mask)
        result = postprocessors(output, target['size'])
        results.append(result)
        targets.append(target)

    evaluator = HICOEvaluator(results, targets)
    evaluator.evaluate()

    print("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaling script', parents=[get_args_parser()])
    args = parser.parse_args()
    run_eval(args)
