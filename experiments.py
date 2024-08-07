import os.path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product
from collections.abc import Iterable

from random import sample

from utils.utils import JobManager, print_args

class CommandBuilder:
    BEST_VALUE = None
    
    def __init__(self, args, hparams_dir=None, random=None):
        self.random = random
        self.default_options = f" -s {args.seed} -r {args.repeats} -o {args.output_dir}"
        if args.project:
            self.default_options += f" --log --log_mode collective --project_name {args.project}"
        self.hparams = None

    def build(self, dataset, model, learning_rate, weight_decay, dropout, prune=False, prune_fraction=None, quat=False):
        cmd_list = []
        configs = self.product_dict(
            dataset=self.get_list(dataset),
            model=self.get_list(model),
            learning_rate=self.get_list(learning_rate),
            weight_decay=self.get_list(weight_decay),
            dropout=self.get_list(dropout),
            prune=self.get_list(prune),
            prune_fraction=self.get_list(prune_fraction),
            quat=self.get_list(quat)
        )

        if self.random:
            configs = sample(configs, self.random)

        for config in configs:
            config = self.fill_best_params(config)
            options = ' '.join([f' --{param} {value} ' for param, value in config.items()])
            command = f'python main.py {options} {self.default_options}'
            cmd_list.append(command)
            
        return cmd_list
            
    def fill_best_params(self, config):
        if self.hparams:
            best_params = self.hparams.get(
                dataset=config['dataset'], 
                model=config['model'],
            )
            
            for param, value in config.items():
                if value == self.BEST_VALUE:
                    config[param] = best_params[param]
                
        return config

    @staticmethod
    def get_list(param):
        if not isinstance(param, Iterable) or isinstance(param, str):
            param = [param]
        return param
    
    @staticmethod
    def product_dict(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))
            
def experiments(args):
    run_cmds = []
    cmdbuilder = CommandBuilder(args=args, hparams_dir='./hparams')
    datasets = ['cora', 'citeseer', 'pubmed']
    
    for dataset in datasets:
        run_cmds += cmdbuilder.build(
            dataset=dataset,
            model=['sage','gat'],
            learning_rate=[0.01, 0.001],
            weight_decay=[5e-4, 0],
            dropout=[0.5, 0],
            prune=False,
            prune_fraction=0
        )
        
    run_cmds = list(set(run_cmds)) # remove duplicates
    return run_cmds

def pruning_based(args):
    run_cmds = []
    cmdbuilder = CommandBuilder(args=args, hparams_dir='./hparams')
    datasets = ['cora', 'citeseer', 'pubmed']
    
    for dataset in datasets:
        run_cmds += cmdbuilder.build(
            dataset=dataset,
            model=['sage', 'gat'],
            learning_rate=[0.01, 0.001],
            weight_decay=[5e-4, 0],
            dropout=[0.5, 0],
            prune=True,
            prune_fraction=[0.3, 0.5, 0.7]
        )
        
    run_cmds = list(set(run_cmds)) # remove duplicates
    return run_cmds

def quaternion(args):
    run_cmds = []
    cmdbuilder = CommandBuilder(args=args, hparams_dir='./hparams')
    datasets = ['cora', 'citeseer', 'pubmed']
    
    for dataset in datasets:
        run_cmds += cmdbuilder.build(
            dataset=dataset,
            model='gcn',
            quat=True,
            learning_rate=[0.01, 0.001],
            weight_decay=[5e-4, 0],
            dropout=[0.5, 0],
            prune=False,
            prune_fraction=0
        )
        
    run_cmds = list(set(run_cmds)) # remove duplicates
    return run_cmds

def quaternion_pruning(args):
    run_cmds = []
    cmdbuilder = CommandBuilder(args=args, hparams_dir='./hparams')
    datasets = ['cora', 'citeseer', 'pubmed']
    
    for dataset in datasets:
        run_cmds += cmdbuilder.build(
            dataset=dataset,
            model='gcn',
            quat=True,
            learning_rate=[0.01, 0.001],
            weight_decay=[5e-4, 0],
            dropout=[0.5, 0],
            prune=True,
            prune_fraction=[0.3, 0.5, 0.7]
        )
        
    run_cmds = list(set(run_cmds)) # remove duplicates
    return run_cmds

def experiment_generator(args):
    run_cmds = []
    
    if args.general:
        run_cmds += experiments(args)
        
    if args.general and args.pruning:
        run_cmds += pruning_based(args)

    if args.quat:
        run_cmds += quaternion(args)
    
    if args.quat and args.pruning:
        run_cmds += quaternion_pruning(args)
        
    return run_cmds

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser, parser_create = JobManager.register_arguments(parser)
    parser.add_argument('-o', '--output_dir', type=str, default='./results', help='Directory to save the results')
    parser_create.add_argument('-p', '--project', type=str, help='Project name')
    parser_create.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser_create.add_argument('-r', '--repeats', type=int, default=10, help='Number of repeats')
    parser.add_argument('--general', action='store_true', help='Run general experiments')
    parser.add_argument('--pruning', action='store_true', help='Run pruning based experiments')
    parser.add_argument('--quat', action='store_true', help='Run quaternion based experiments')
    args = parser.parse_args()
    print_args(args)

    JobManager(args, cmd_generator=experiment_generator).run()

if __name__ == '__main__':
    main()