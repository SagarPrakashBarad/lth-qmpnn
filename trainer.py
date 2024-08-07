import os
import sys
import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch_geometric.loader import DataLoader
from tqdm import trange
import wandb

from prune import prune_model, reset_model

class BaseTrainer:
    def __init__(
            self,
            optimizer:     dict(help='Optimizer', option='-o', choices=['sgd', 'adam']) = 'adam',
            max_epochs:    dict(help='Maximum number of epochs') = 1000,
            learning_rate: dict(help='Learning rate') = 0.005,
            weight_decay:  dict(help='Weight decay') = 5e-4,
            patience:      dict(help='Patience for early stopping') = 200,
            device:        dict(help='Device to run the training') = 'cuda',
            prune:         dict(help='Prune the model') = False,
            prune_fraction:dict(help='Number of iterations to prune', type=float) = 0.5,
            quat:          dict(help='Use quaternion representation') = False,
    ):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.prune = prune
        self.prune_fraction = prune_fraction
        self.prune_iter = 20
        self.model = None
        self.quat = quat
        
    def configure_optimizers(self):
        if self.optimizer == 'sgd':
            return SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')
        
    def fit(self, model, data):
        pass
    
    def fit_with_pruning(self, model, data):
        pass
    
    def _train(self, data, optimizer):
        # self.model.train()
        # optimizer.zero_grad()
        # loss, metrics = self.model.training_step(data)
        # loss.backward()
        # optimizer.step()
        # return metrics
        pass
    
    @torch.no_grad()
    def _validation(self, data):
        # self.model.eval()
        # return self.model.validation_step(data)
        pass
        

class NodeClassifierTrainer(BaseTrainer):
    def __init__(self, 
                 optimizer: dict(help='Optimizer', option='-o', choices=['sgd', 'adam']) = 'adam', 
                 max_epochs: dict(help='Maximum number of epochs') = 1000, 
                 learning_rate: dict(help='Learning rate') = 0.005, 
                 weight_decay: dict(help='Weight decay') = 0.0005, 
                 patience: dict(help='Patience for early stopping') = 200, 
                 device: dict(help='Device to run the training') = 'cuda', 
                 prune: dict(help='Prune the model') = False, 
                 prune_fraction: dict(help='Number of iterations to prune', type=float) = 0.5, 
                 quat: dict(help='Use quaternion representation') = False,
                 task: dict(help='Task to perform', choices=['node', 'graph']) = 'node'):
        super().__init__(optimizer, max_epochs, learning_rate, weight_decay, patience, device, prune, prune_fraction, quat)
        self.task = task
        
    def fit(self, model, data, args):
        self.model = model.to(self.device)
        data = data.to(self.device)
        optimizer = self.configure_optimizers()

        num_epochs_without_improvement = 0
        best_metrics = None

        epoch_progbar = trange(1, self.max_epochs+1, desc='Epochs', leave=False, position=1, file=sys.stdout)
        for epoch in epoch_progbar:
            metrics = {'epoch': epoch}
            train_metrics = self._train(data, optimizer)
            metrics.update(train_metrics)
            
            val_metrics = self._validation(data)
            metrics.update(val_metrics)
            
            test_metrics = self._test(data)
            metrics.update(test_metrics)
            
            
            if best_metrics is None or (
                metrics['val/loss'] < best_metrics['val/loss'] and
                best_metrics['val/acc'] < metrics['val/acc'] and 
                best_metrics['train/acc'] < metrics['train/acc']
            ):
                best_metrics = metrics
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                if num_epochs_without_improvement >= self.patience > 0:
                    break

            # display metrics on progress bar
            epoch_progbar.set_postfix(metrics)
            wandb.log(metrics)

        return best_metrics['test/acc']
    
    def fit_with_pruning(self, model, data, args):
        model_name = model.gnn.__class__.__name__
        dataset_name = data.name
        
        prune_run_metrics = {}
        
        dir_name = f'saved_models'
        os.makedirs(dir_name, exist_ok=True)
        
        # save original model
        torch.save(model, f'{dir_name}/{model_name}_{dataset_name}_orig.pt')
        
        l, a, acc = [], [], []
        for prune_i in trange(self.prune_iter):
            # resetting the model
            #model.apply(reset_model)
            
            # training the model
            best_metrics = self.fit(model, data, args)
            wandb.log({"best_metrics" : best_metrics}, step = prune_i + 1)
            acc.append(best_metrics)
            
            # save the model
            torch.save(model, f'{dir_name}/{model_name}_{dataset_name}_{prune_i}.pt')
            
            # pruning the model
            prune_model(model, fraction=self.prune_fraction)
            
            # resetting the model
            reset_model(model, torch.load(f'{dir_name}/{model_name}_{dataset_name}_orig.pt', map_location = self.device))
            
        print(acc)
        
        return acc

    def _train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        loss, train_acc = self.model.training_step(data)
        loss.backward()
        optimizer.step()
        return {'train/loss': loss.item(), 'train/acc': train_acc}

    @torch.no_grad()
    def _validation(self, data):
        self.model.eval()
        loss, val_acc = self.model.validation_step(data)
        return {'val/loss': loss.item(), 'val/acc': val_acc}
    
    def _test(self, data):
        self.model.eval()
        loss, test_acc = self.model.validation_step(data)
        return {'test/acc': test_acc}

class GraphClassifierTrainer(BaseTrainer):
    def __init__(self, 
                 optimizer: dict(help='Optimizer', option='-o', choices=['sgd', 'adam']) = 'adam', 
                 max_epochs: dict(help='Maximum number of epochs') = 1000, 
                 learning_rate: dict(help='Learning rate') = 0.005, 
                 weight_decay: dict(help='Weight decay') = 0.0005, 
                 patience: dict(help='Patience for early stopping') = 200, 
                 device: dict(help='Device to run the training') = 'cuda', 
                 prune: dict(help='Prune the model') = False, 
                 prune_fraction: dict(help='Number of iterations to prune', type=float) = 0.5, 
                 quat: dict(help='Use quaternion representation') = False,
                 task: dict(help='Task to perform', choices=['node', 'graph']) = 'graph'):
        super().__init__(optimizer, max_epochs, learning_rate, weight_decay, patience, device, prune, prune_fraction, quat)
        self.task = task


    def fit(self, model, data, args):
        self.model = model.to(self.device)
        optimizer = self.configure_optimizers()

        num_epochs_without_improvement = 0
        best_metrics = None

        epoch_progbar = trange(1, self.max_epochs+1, desc='Epochs', leave=False, position=1, file=sys.stdout)
        
        for epoch in epoch_progbar:
            metrics = {'epoch': epoch}
            #print(len(data.adj_t))
            train_metrics = self._train(data, optimizer)
            metrics.update(train_metrics)

            val_metrics = self._validation(data)
            metrics.update(val_metrics)
            
            test_metrics = self._test(data)
            metrics.update(test_metrics)

            if best_metrics is None or (
                metrics['val/loss'] < best_metrics['val/loss'] and
                best_metrics['val/acc'] < metrics['val/acc'] and 
                best_metrics['train/acc'] < metrics['train/acc']
            ):
                best_metrics = metrics
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                if num_epochs_without_improvement >= self.patience > 0:
                    break
            
            # display metrics on progress bar
            epoch_progbar.set_postfix(metrics)
            
            
            wandb.log(metrics)

        return best_metrics['test/acc']
    
    def fit_with_pruning(self, model, data, args):
        model_name = model.gnn.__class__.__name__
        dataset_name = data.name
        
        prune_run_metrics = {}
        
        dir_name = f'saved_models'
        os.makedirs(dir_name, exist_ok=True)
        
        # save original model
        torch.save(model, f'{dir_name}/{model_name}_{dataset_name}_orig.pt')
        
        l, a, acc = [], [], []
        for prune_i in trange(self.prune_iter):
            # resetting the model
            #model.apply(reset_model)
            
            # training the model
            best_metrics = self.fit(model, data, args)
            # print(f"Prune iteration: {prune_i}, best_metrics: {best_metrics}")
            acc.append(best_metrics)
            
            # save the model
            torch.save(model, f'{dir_name}/{model_name}_{dataset_name}_{prune_i}.pt')
            
            # pruning the model
            prune_model(model, fraction=self.prune_fraction)
            
            # resetting the model
            reset_model(model, torch.load(f'{dir_name}/{model_name}_{dataset_name}_orig.pt', map_location = self.device))
        
        return acc

    def _train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        metrics = self.model.training_step(data, optimizer)
        return metrics

    @torch.no_grad()
    def _validation(self, data):
        self.model.eval()
        metrics = self.model.validation_step(data)
        return metrics
    
    @torch.no_grad()
    def _test(self, data):
        metrics = {}
        metrics_val = self.model.validation_step(data)
        metrics['test/acc'] = metrics_val['val/acc']
        return metrics
    
class LinkPredictorTrainer(BaseTrainer):
    def __init__(self, 
                 optimizer: dict(help='Optimizer', option='-o', choices=['sgd', 'adam']) = 'adam', 
                 max_epochs: dict(help='Maximum number of epochs') = 1000, 
                 learning_rate: dict(help='Learning rate') = 0.005, 
                 weight_decay: dict(help='Weight decay') = 0.0005, 
                 patience: dict(help='Patience for early stopping') = 200, 
                 device: dict(help='Device to run the training') = 'cuda', 
                 prune: dict(help='Prune the model') = False, 
                 prune_fraction: dict(help='Number of iterations to prune', type=float) = 0.5, 
                 quat: dict(help='Use quaternion representation') = False,
                 task: dict(help='Task to perform', choices=['node', 'graph', 'link']) = 'link'):
        super().__init__(optimizer, max_epochs, learning_rate, weight_decay, patience, device, prune, prune_fraction, quat)
        self.task = task

    def fit(self, model, data, args):
        
        self.model = model.to(self.device)
        optimizer = self.configure_optimizers()

        num_epochs_without_improvement = 0
        best_metrics = None

        epoch_progbar = trange(1, self.max_epochs+1, desc='Epochs', leave=False, position=1, file=sys.stdout)
        
        for epoch in epoch_progbar:
            metrics = {'epoch': epoch}
            train_data, val_data, test_data = data[0]
            train_metrics = self._train(train_data, optimizer)
            metrics.update(train_metrics)

            val_metrics = self._validation(val_data)
            metrics.update(val_metrics)
            
            test_metrics = self._test(test_data)
            metrics.update(test_metrics)

            if best_metrics is None or (
                metrics['val/loss'] < best_metrics['val/loss'] and
                best_metrics['val/roc'] < metrics['val/roc'] and 
                best_metrics['train/roc'] < metrics['train/roc']
            ):
                best_metrics = metrics
                num_epochs_without_improvement = 0
            else:
                num_epochs_without_improvement += 1
                if num_epochs_without_improvement >= self.patience > 0:
                    break

            # display metrics on progress bar
            epoch_progbar.set_postfix(metrics)
            if True or args.log:
                wandb.log(metrics)

        return best_metrics['test/roc']

    def fit_with_pruning(self, model, data, args):
        model_name = model.gnn.__class__.__name__
        dataset_name = data.name
        
        prune_run_metrics = {}
        
        dir_name = f'saved_models'
        os.makedirs(dir_name, exist_ok=True)
        
        # save original model
        torch.save(model, f'{dir_name}/{model_name}_{dataset_name}_orig.pt')
        
        l, a, acc = [], [], []
        for prune_i in trange(self.prune_iter):
            # resetting the model
            #model.apply(reset_model)
            
            # training the model
            best_metrics = self.fit(model, data, args)
            print(f"Prune iteration: {prune_i}, best_metrics:{best_metrics}")
            acc.append(best_metrics)
            
            # save the model
            torch.save(model, f'{dir_name}/{model_name}_{dataset_name}_{prune_i}.pt')
            
            # pruning the model
            prune_model(model, fraction=self.prune_fraction)
            
            # resetting the model
            reset_model(model, torch.load(f'{dir_name}/{model_name}_{dataset_name}_orig.pt', map_location = self.device))

        return acc
    
    def _train(self, data, optimizer):
        self.model.train()
        optimizer.zero_grad()
        loss, roc_auc = self.model.training_step(data, optimizer)
        return {'train/loss': loss.item(), 'train/roc': roc_auc}
    
    def _validation(self, data):
        self.model.eval()
        loss, roc_auc = self.model.validation_step(data)
        return {'val/loss': loss.item(), 'val/roc': roc_auc}
    
    def _test(self, data):
        self.model.eval()
        loss, roc_auc = self.model.validation_step(data)
        return {'test/roc': roc_auc}
    
