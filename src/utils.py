import torch
import os
import random
import numpy as np


def save_load_name(args, names):
    model_name = names['model_name']
    return f'{args.dataset}_{model_name}'


def save_model(args, model, names):
    name = save_load_name(args, names)
    torch.save(model, f'{args.model_path}/{name}.pt')
    print(f"Saved model at {args.model_path}/{name}.pt!")


def load_model(args, names):
    name = save_load_name(args, names)
    print(f"Loading model at {args.model_path}/{name}.pt!")
    model = torch.load(f'{args.model_path}/{name}.pt', weights_only=False)
    return model


def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)


def transfer_models(new_model, pretrained_models):
    pretrained_t_model, pretrained_a_model, pretrained_v_model = pretrained_models
    new_dict = new_model.state_dict()

    t_model = torch.load(pretrained_t_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_t_dict = t_model.state_dict()
    t_proj_state_dict = {}
    t_enc_state_dict = {}
    for k, v in pretrain_t_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.0'
            new_k = '.'.join(k_list)
            t_proj_state_dict[new_k] = v
        else:
            t_enc_state_dict[k] = v
    new_dict.update(t_proj_state_dict)
    new_dict.update(t_enc_state_dict)

    a_model = torch.load(pretrained_a_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_a_dict = a_model.state_dict()
    a_proj_state_dict = {}
    a_enc_state_dict = {}
    for k, v in pretrain_a_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.1'
            new_k = '.'.join(k_list)
            a_proj_state_dict[new_k] = v
        else:
            a_enc_state_dict[k] = v
    new_dict.update(a_proj_state_dict)
    new_dict.update(a_enc_state_dict)

    v_model = torch.load(pretrained_v_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_v_dict = v_model.state_dict()
    v_proj_state_dict = {}
    v_enc_state_dict = {}
    for k, v in pretrain_v_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.2'
            new_k = '.'.join(k_list)
            v_proj_state_dict[new_k] = v
        else:
            v_enc_state_dict[k] = v
    new_dict.update(v_proj_state_dict)
    new_dict.update(v_enc_state_dict)

    new_model.load_state_dict(new_dict)

    return new_model

def transfer_model(new_model, pretrained_model):
    new_dict = new_model.state_dict()
    pretrain_dict = pretrained_model.state_dict()
    enc_state_dict = {}
    for k, v in pretrain_dict.items():
        if k not in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ] and k in new_dict:
            # print('Load weight: ', k)
            enc_state_dict[k] = v
    new_dict.update(enc_state_dict)
    new_model.load_state_dict(new_dict)
    # print('Transfer weights done!')

    # for name, param in new_model.named_parameters():
    #     if name in enc_state_dict:
    #         param.requires_grad = False

    return new_model, enc_state_dict