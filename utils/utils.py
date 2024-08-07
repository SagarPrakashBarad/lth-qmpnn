import os
import time
from argparse import ArgumentTypeError
import inspect
import functools
from subprocess import check_call

from tqdm import tqdm

def measure_runtime(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'\nTotal time spent in {str(func.__name__)}: {end-start:.2f} seconds\n\n')
        return result
    return wrapper

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def add_parameters_as_arguments(function, parser):
    if inspect.isclass(function):
        function = function.__init__
    parameters = inspect.signature(function).parameters
    for param_name, param_obj in parameters.items():
        if param_obj.annotation is not inspect.Parameter.empty:
            arg_info = param_obj.annotation
            arg_info['default'] = param_obj.default
            arg_info['dest'] = param_name
            arg_info['type'] = arg_info.get('type', type(param_obj.default))

            if arg_info['type'] is bool:
                arg_info['type'] = str2bool
                arg_info['nargs'] = '?'
                arg_info['const'] = True

            if 'choices' in arg_info:
                arg_info['help'] = arg_info.get('help', '') + f" (choices: {', '.join(arg_info['choices'])})"
                arg_info['metavar'] = param_name.upper()

            options = {f'--{param_name}', f'--{param_name.replace("_", "-")}'}
            custom_options = arg_info.pop('option', [])
            custom_options = [custom_options] if isinstance(custom_options, str) else custom_options
            options.update(custom_options)
            options = sorted(sorted(list(options)), key=len)
            parser.add_argument(*options, **arg_info)


def strip_unexpected_kwargs(func, kwargs):
    signature = inspect.signature(func)
    parameters = signature.parameters

    # Check if the function has kwargs
    for name, param in parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
        
    kwargs = {arg: value for arg, value in kwargs.items() if arg in parameters}
    return kwargs

def from_args(func, ns, *args, **kwargs):
    return func(*args, **strip_unexpected_kwargs(func, vars(ns)), **kwargs)

class JobManager:
    def __init__(self, args, cmd_generator=None):
        self.args = args
        self.name = args.name
        self.command = args.command
        self.jobs_dir = args.jobs_dir
        self.cmd_generator = cmd_generator
        
    def run(self):
        if self.command == 'create':
            self.create()
        elif self.command == 'exec':
            self.exec()
            
    def create(self):
        os.makedirs(self.jobs_dir, exist_ok=True)
        run_cms = self.cmd_generator(self.args)
        
        with open(os.path.join(self.jobs_dir, f'{self.name}.jobs'), 'w') as file:
            for run in tqdm(run_cms):
                file.write(run + '\n')

        print('job file created:', os.path.join(self.jobs_dir, f'{self.name}.jobs'))

    def exec(self):
        with open(os.path.join(self.jobs_dir, f'{self.name}.jobs'), 'r') as jobs_file:
            job_list = jobs_file.read().splitlines()

        total_jobs = len(job_list)
        job_counter = 0 
        if self.args.all:
            for cmd in job_list:
                job_counter += 1
                print(f'Running job {job_counter}/{total_jobs}')
                check_call(cmd.split())
        else:
            check_call(job_list[self.args.id-1].split())
            
    @staticmethod
    def register_arguments(parser, default_jobs_dir='./jobs'):
        parser.add_argument('-n', '--name', type=str, required=True, help='Name of the job file')
        parser.add_argument('-j', '--jobs_dir', type=str, default=default_jobs_dir, help='Directory to save the job file')
        command_subparser = parser.add_subparsers(dest='command')
        
        parser_create = command_subparser.add_parser('create')
        
        parser_exec = command_subparser.add_parser('exec')
        parser_exec.add_argument('-i', '--id', type=int, help='Job id to execute')
        parser_exec.add_argument('-a', '--all', action='store_true', help='Execute all jobs in the file')
        
        return parser, parser_create
        
             
def print_args(args):
    message = [f'{name}: {value}' for name, value in vars(args).items()]
    print(', '.join(message)+'\n')