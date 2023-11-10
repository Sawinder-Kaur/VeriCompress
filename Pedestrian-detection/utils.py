import torch.nn.utils.prune as prune

import torch_pruning as tp
import torch
import torch.nn as nn
import numpy as np
from feedforward import cnn_7layer,cnn_16layer,cnn_4layer

def count_parameters(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    non_zero_parameters = 0
    for i in layer_names:
        if "weight" in i:
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights += np.sum(np.ones_like(weights))
            non_zero_parameters += np.count_nonzero(weights)
    
    print("Total number of parameters : {}".format(int(total_weights)))
    print("Total number of non-zero parameters : {}".format(non_zero_parameters))
    print("Fraction of paramaters which are non-zero : {}".format(non_zero_parameters/total_weights))

def count_parameters_per_layer(model):
    mydict = model.state_dict()
    layer_names = list(mydict)
    total_weights = 0
    non_zero_parameters = 0
    layer_num = 1
    for i in layer_names:
        if "weight" in i:
            #print(i)
            weights = np.abs((model.state_dict()[i]).detach().cpu().numpy())
            total_weights = np.sum(np.ones_like(weights))
            non_zero_parameters = np.count_nonzero(weights)
            print("Pruning at layer {}: {}".format(layer_num,(total_weights-non_zero_parameters)/total_weights))
            layer_num += 1


def compute_mask(module, score, prune_ratio):
    split_val = torch.quantile(score,prune_ratio)

    struct_mask = torch.where(score <= split_val, 0.0,1.0)
    fine_mask_l = []
    
    if isinstance(module,nn.Linear):
        weight = torch.transpose(module.weight,0,1)
    else:
        weight = module.weight
    
    for mask, m in zip(struct_mask, weight):
        #print(mask)
        if mask == 0: 
            fine_mask_l.append(torch.zeros_like(m))
        else:
            fine_mask_l.append(torch.ones_like(m))
    
    fine_mask = torch.stack(fine_mask_l)

    return fine_mask,struct_mask


def deactivate_elements(model,parameters_to_prune,device = "cuda"):
    
    importance_score = []
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            importance_score.append(module.weight)
        elif isinstance(module, nn.Linear):
            importance_score.append(torch.transpose(module.weight,0,1))
        
    i = 0
    ll_num = 0
    fine_mask = []
    struct_mask = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d): 
            #print(module.weight.size())
            kernel_score = []
            for i_score in importance_score[i]:
                kernel_score.append(torch.norm(i_score,2).item())    
            kernel_score = torch.tensor(kernel_score).to(device)
            fine_mask_l, struct_mask_l = compute_mask(module, kernel_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            struct_mask.append(struct_mask_l)
            i += 1
        elif isinstance(module, nn.Linear):
            #print(module.weight.size())
            node_score = []
            for i_score in importance_score[i]:
                node_score.append(torch.norm(i_score,2).item())  
            node_score = torch.tensor(node_score).to(device)
            
            fine_mask_l, struct_mask_l = compute_mask(module, node_score, parameters_to_prune[i])
            fine_mask.append(fine_mask_l)
            if ll_num != 0: 
                struct_mask.append(struct_mask_l)
            ll_num += 1
            i += 1
            
    """
    for m in fine_mask:
        print(m.size())
    """
    #torch.where(importance_score3[2] < split_val, 0.0, importance_score3[2].double())
    model = apply_mask(model, fine_mask,struct_mask)
    count_parameters(model)
    #print(struct_mask)
    
    return struct_mask

def apply_mask(model, model_mask,struct_mask = None):
    mydict = model.state_dict()
    layer_names = list(mydict)
    i = 0
    if "weight" in layer_names[0]: 
        w_ln = 0
        b_ln = 1
    else: 
        w_ln = 1
        b_ln = 0
    for module in model.modules():
        
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Linear): 
                mask = torch.transpose(model_mask[i],0,1)
            else: 
                mask = model_mask[i]
            model.state_dict()[layer_names[w_ln]].copy_(model.state_dict()[layer_names[w_ln]] * mask)
            if len(model.state_dict()[layer_names[b_ln]]) != 2 and struct_mask != None: 
                model.state_dict()[layer_names[b_ln]].copy_(model.state_dict()[layer_names[b_ln]] * struct_mask[i])
            i = i + 1
            w_ln = w_ln+2
            b_ln = b_ln+2
        elif isinstance(module, nn.BatchNorm2d):
            w_ln = w_ln+5
            b_ln = b_ln+5
    return model




def remove_redundant_nodes(model,example_inputs,parameter_budget):
    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = []
    total_weights = count_parameters(model)
    compress_amount = parameter_budget/total_weights
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d)or isinstance(m, torch.nn.BatchNorm2d) or (isinstance(m, torch.nn.Linear) and m.out_features == 2):
            ignored_layers.append(m) # DO NOT prune the final classifier!
    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=compress_amount, 
        ignored_layers=ignored_layers,
    )
    for i in range(iterative_steps):
        pruner.step()
    return model


def prune_model_structured_erdos_renyi_kernel(model,prune_amount):
    
    """
    if prune_amount > 0.98: 
        print("Pruning amount too high for structured pruning")
        exit()
    print("Global Pruning Amount : {}%".format(prune_amount*100))
    """
    parameters_to_prune = []
    for module in  model.modules():
        if isinstance(module, nn.Conv2d):
            #print(module.kernel_size )
            scale = 1.0 - (module.in_channels + module.out_channels + module.kernel_size[0] + module.kernel_size[1] )/(module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1])
            #print(scale)
            parameters_to_prune.append(scale * prune_amount)
            #parameters_to_prune.append(scale * prune_amount/2.0)
        elif (isinstance(module, nn.Linear) and module.out_features == 2):
            parameters_to_prune.append(prune_amount/2.0)
        elif isinstance(module, nn.Linear) :
            scale = 1.0 - (module.in_features + module.out_features)/(module.in_features * module.out_features)
            #print(scale)
            if prune_amount < 0.98 : parameters_to_prune.append(scale * prune_amount + 0.02)
            else: parameters_to_prune.append(scale * prune_amount)
    
    return parameters_to_prune


def deactivate(model, args):
    dummy_input = torch.randn(1, 3, 140, 140)
    model_mask = cnn_7layer(in_ch=3, in_dim=140)
    
    total_weights = count_parameters(model)
    compress_amount = args.parameter_budget/total_weights
    parameters_to_prune = prune_model_structured_erdos_renyi_kernel(model,compress_amount)
    
    element_states = deactivate_elements(model,parameters_to_prune,device = args.device)
       
    return element_states