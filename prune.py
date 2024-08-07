import torch
import torch.nn.utils.prune as prune
from torch import nn
import torch_geometric
from qgnn import *

def prune_model(model, fraction):
    if isinstance(model, nn.Module):
        prune_method = prune.l1_unstructured
    else:
        raise ValueError('The provided model is not a valid nn.Module')
    
    if fraction == 0:
        return model
    if fraction < 0 or fraction > 1:
        raise ValueError('Pruning fraction should be in [0, 1]')
            
    for module_name, module in model.named_modules():
        if isinstance(module, torch_geometric.nn.GATConv):
            prune_method(module, name='lin_src', amount=fraction)
        elif isinstance(module, torch_geometric.nn.GCNConv):
            prune_method(module.lin, name='weight', amount=fraction)
        elif isinstance(module, torch_geometric.nn.SAGEConv):
            prune_method(module.lin_l, name='weight', amount=fraction)
            prune_method(module.lin_r, name='weight', amount=fraction)
        elif isinstance(module, QGCNLayer) or isinstance(module, QGCNLayer_v2):
            prune_method(module, name='weight', amount=fraction)            
        elif isinstance(module, QGATLayer) or isinstance(module, QGATLayer_v2):
            prune_method(module, name = 'W', amount=fraction)
            prune_method(module, name = 'a', amount=fraction)
        elif isinstance(module, QGATMultiHeadLayer):
             for i in range(module.num_heads):
                prune_method(module.W_heads[i], name='weight', amount=fraction)
                prune_method(module.a_heads[i], name='weight', amount=fraction)
        elif isinstance(module, QSAGELayer) or isinstance(module, QSAGELayer_v2):
            prune_method(module, name = 'weight', amount = fraction)
    return model


def reset_model(model, model_orig):
    for (module_name, module), (orig_module_name, orig_module) in zip(model.named_modules(), model_orig.named_modules()):
        if isinstance(module, torch_geometric.nn.GATConv):
            mask = module.lin_src.weight_mask
            prune.custom_from_mask(orig_module.lin_src, name='weight', mask=mask)
        elif isinstance(module, torch_geometric.nn.GCNConv):
            mask = module.lin.weight_mask
            prune.custom_from_mask(orig_module.lin, name='weight', mask=mask)
        elif isinstance(module, torch_geometric.nn.SAGEConv):
            # For SAGEConv, prune both lin_l and lin_r weights
            mask_l = module.lin_l.weight_mask if hasattr(module.lin_l, 'weight_mask') else None
            mask_r = module.lin_r.weight_mask if hasattr(module.lin_r, 'weight_mask') else None
            if mask_l is not None:
                prune.custom_from_mask(orig_module.lin_l, name='weight', mask=mask_l)
            if mask_r is not None:
                prune.custom_from_mask(orig_module.lin_r, name='weight', mask=mask_r)
        elif isinstance(module, QGCNLayer) or isinstance(module, QGCNLayer_v2):
            mask = module.weight_mask
            prune.custom_from_mask(orig_module, name='weight', mask=mask)
        elif isinstance(module, QSAGELayer) and isinstance(module, QSAGELayer_v2):
            mask = module.weight_mask
            prune.custom_from_mask(orig_module, name='weight', mask=mask)
            
        elif isinstance(module, QGATLayer):
            # Reset weights for QGATLayer
            mask_W = module.W.weight_mask if hasattr(module.W, 'weight_mask') else None
            mask_a = module.a.weight_mask if hasattr(module.a, 'weight_mask') else None
            if mask_W is not None:
                prune.custom_from_mask(orig_module.W, name='weight', mask=mask_W)
            if mask_a is not None:
                prune.custom_from_mask(orig_module.a, name='weight', mask=mask_a)
        elif isinstance(module, QGATMultiHeadLayer):
            # Reset weights for QGATMultiHeadLayer
            for i in range(module.num_heads):
                mask_W_head = module.W_heads[i].weight_mask if hasattr(module.W_heads[i], 'weight_mask') else None
                mask_a_head = module.a_heads[i].weight_mask if hasattr(module.a_heads[i], 'weight_mask') else None
                if mask_W_head is not None:
                    prune.custom_from_mask(orig_module.W_heads[i], name='weight', mask=mask_W_head)
                if mask_a_head is not None:
                    prune.custom_from_mask(orig_module.a_heads[i], name='weight', mask=mask_a_head)

    return model
   