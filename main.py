import os
import sys
import traceback
import uuid
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import numpy as np
import pandas as pd
import wandb
from tqdm.auto import tqdm

import torch

from datasets import load_dataset, preprocess_batch
from models import NodeClassifier, GraphClassifier, LinkPredictor
from trainer import BaseTrainer, NodeClassifierTrainer, GraphClassifierTrainer, LinkPredictorTrainer

from utils.utils import JobManager, print_args, measure_runtime, from_args, add_parameters_as_arguments

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@measure_runtime
def run(args):
    data = from_args(load_dataset, args)
    print('Dataset loaded')
    #print(data.x)
    test_acc = []
    run_metrics = {}
    run_id = str(uuid.uuid1())

    progbar = tqdm(range(args.repeats), file=sys.stdout)
    for version in progbar:

        if args.quat:
            print('Quat is enabled')
            
        # define model
        if args.task == 'node':
            model = from_args(NodeClassifier, args, input_dim=data.num_features, num_classes=data.num_mod_classes)
            trainer = from_args(NodeClassifierTrainer, args)
        elif args.task == 'graph':
            model = from_args(GraphClassifier, args, input_dim=data.x.shape[1], num_classes=data.num_mod_classes)
            trainer = from_args(GraphClassifierTrainer, args)
        elif args.task == 'link':
            model = from_args(LinkPredictor, args, input_dim=data.x.shape[1])
            trainer = from_args(LinkPredictorTrainer, args)
            
        # if args.log:
        if True:
            wandb.init(
                project='QGNNs',
                # project = args.project,
                name=f"{args.model}_{args.task}_{'quat' if args.quat else 'real'}_{'pruned' if args.prune else 'unpruned'}",
                config=args
            )
    
        if args.prune:
            best_metrics = trainer.fit_with_pruning(model, data, args)
        else:
            best_metrics = trainer.fit(model, data, args)
        
        print(best_metrics)
        
        # Define the directory where you want to save the metrics
        runs_directory = 'runs'
        os.makedirs(runs_directory, exist_ok=True)
        filename = f"{args.model}_{args.dataset}_{args.task}_{'quat' if args.quat else 'real'}_{'pruned' if args.prune else 'unpruned'}_metrics.txt"
        full_path = os.path.join(runs_directory, filename)
        num_params = sum(p.numel() for p in model.parameters()) / 1e6  

        with open(full_path, 'a') as f:
            f.write(str(best_metrics) + '\n' + str(num_params))
        # process results
        # for metric, value in best_metrics.items():
        #     run_metrics[metric] = run_metrics.get(metric, []) + [value]

        # test_acc.append(best_metrics['test/acc'])
        # test_acc.append(best_metrics)
        # progbar.set_postfix({'last_test_acc': test_acc[-1], 'avg_test_acc': np.mean(test_acc)}
    # os.makedirs(args.output_dir, exist_ok=True)
    # df_results = pd.DataFrame()
    # df_results['test_acc'] = test_acc
    # df_results['version'] = run_id
    # df_results.reset_index(inplace=True)
    # df_results['Name'] = run_id
    # for arg_name, arg_val in vars(args).items():
    #     df_results[arg_name] = [arg_val] * len(test_acc)
    # df_results.to_csv(os.path.join(args.output_dir, f'{run_id}.csv'), index=False)

def main():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')
    
    

    # general args
    # init_parser.add_argument('-q', '--quaternion', dest='quat', action='store_true', help='Enable quaternion')

    # dataset args
    group_dataset = init_parser.add_argument_group('Dataset arguments')
    add_parameters_as_arguments(load_dataset, group_dataset)

    # model args
    group_model = init_parser.add_argument_group('Model arguments')
    add_parameters_as_arguments(NodeClassifier, group_model)
    add_parameters_as_arguments(GraphClassifier, group_model)
    add_parameters_as_arguments(LinkPredictor, group_model)

    # trainer args
    group_trainer = init_parser.add_argument_group('Trainer arguments')
    add_parameters_as_arguments(BaseTrainer, group_trainer)
    group_trainer.add_argument('--device', help='Device to run the training', choices=['cpu', 'cuda:0', 'cuda:1'], default='cuda:0')

    # experiment args
    group_expr = init_parser.add_argument_group('Experiment arguments')
    group_expr.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    group_expr.add_argument('-r', '--repeats', type=int, default=10, help='Number of repeats')
    group_expr.add_argument('-o', '--output_dir', type=str, default='./results', help='Directory to save the results')

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    print_args(args)
    args.cmd = ' '.join(sys.argv) # store the command

    if args.seed:
        seed_everything(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available, falling back to CPU')
        args.device = 'cpu'

    try:
        run(args)
    except KeyboardInterrupt:
        print('Graceful Shutdown...')

if __name__ == '__main__':
    main()