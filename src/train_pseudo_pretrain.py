import pickle

from torch import nn
import torch.nn.functional as F
from src import models
from src.utils import *
import torch.optim as optim
import time
from src.eval_metrics import *


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, dataloaders, device):
    t_model = models.TextModel(hyp_params)
    a_model = models.AudioModel(hyp_params)
    v_model = models.VisionModel(hyp_params)
    ema_t_model = models.TextModel(hyp_params).eval()
    ema_a_model = models.AudioModel(hyp_params).eval()
    ema_v_model = models.VisionModel(hyp_params).eval()
    ema_t_model.load_state_dict(t_model.state_dict())
    ema_a_model.load_state_dict(a_model.state_dict())
    ema_v_model.load_state_dict(v_model.state_dict())

    if hyp_params.use_cuda:
        t_model = t_model.to(device)
        a_model = a_model.to(device)
        v_model = v_model.to(device)
        ema_t_model = ema_t_model.to(device)
        ema_a_model = ema_a_model.to(device)
        ema_v_model = ema_v_model.to(device)

    bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_params = list(t_model.text_model.named_parameters())
    bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
    bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
    model_params_other = [p for n, p in list(t_model.named_parameters()) if 'text_model' not in n]
    optimizer_grouped_parameters = [
        {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
        {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
        {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
    ]
    t_optimizer = optim.Adam(optimizer_grouped_parameters)
    a_optimizer = optim.Adam(a_model.parameters())
    v_optimizer = optim.Adam(v_model.parameters())
    task_criterion = getattr(nn, hyp_params.criterion)()
    contrastive_criterion = models.SupConLoss(temperature=hyp_params.pretrain_temperature)
    settings = {'t_model': t_model,
                'a_model': a_model,
                'v_model': v_model,
                'ema_t_model': ema_t_model,
                'ema_a_model': ema_a_model,
                'ema_v_model': ema_v_model,
                't_optimizer': t_optimizer,
                'a_optimizer': a_optimizer,
                'v_optimizer': v_optimizer,
                'task_criterion': task_criterion,
                'contrastive_criterion': contrastive_criterion}
    return train_model(settings, hyp_params, dataloaders, device)


####################################################################
#
# Training and evaluation scripts
#
####################################################################
def train_model(settings, hyp_params, dataloaders, device):
    t_model = settings['t_model']
    a_model = settings['a_model']
    v_model = settings['v_model']
    ema_t_model = settings['ema_t_model']
    ema_a_model = settings['ema_a_model']
    ema_v_model = settings['ema_v_model']
    t_optimizer = settings['t_optimizer']
    a_optimizer = settings['a_optimizer']
    v_optimizer = settings['v_optimizer']
    task_criterion = settings['task_criterion']
    contrastive_criterion = settings['contrastive_criterion']
    train_pseudo_labels = {}
    valid_pseudo_labels = {}
    test_pseudo_labels = {}
    init_t_momentum = hyp_params.init_t_momentum
    init_a_momentum = hyp_params.init_a_momentum
    init_v_momentum = hyp_params.init_v_momentum
    init_ema_t_model_momentum = hyp_params.init_ema_t_model_momentum
    init_ema_a_model_momentum = hyp_params.init_ema_a_model_momentum
    init_ema_v_model_momentum = hyp_params.init_ema_v_model_momentum
    def train(models, ema_models, optimizers, task_criterion, contrastive_criterion, epoch):
        t_model, a_model, v_model = models
        ema_t_model, ema_a_model, ema_v_model = ema_models
        t_optimizer, a_optimizer, v_optimizer = optimizers
        t_epoch_loss = 0
        a_epoch_loss = 0
        v_epoch_loss = 0
        c_epoch_loss = 0
        t_model.train()
        a_model.train()
        v_model.train()

        results = {
            'T': [],
            'A': [],
            'V': []
        }
        truths = {
            'T': [],
            'A': [],
            'V': []
        }

        for batch_step, batch_data in enumerate(dataloaders['train']):
            batch_step += (epoch - 1) * len(dataloaders['train']) + 1
            text = batch_data['text'].to(device)
            audio = batch_data['audio'].to(device)
            vision = batch_data['vision'].to(device)
            video_id = batch_data['id']
            labels_mm = batch_data['labels']['M'].to(device)
            labels_tt = batch_data['labels']['T'].to(device)
            labels_aa = batch_data['labels']['A'].to(device)
            labels_vv = batch_data['labels']['V'].to(device)
            one_hot_labels_m = F.one_hot(labels_mm.view(-1).long(), num_classes=3)

            batch_size = text.size(0)
            t_model.zero_grad()
            a_model.zero_grad()
            v_model.zero_grad()

            pseudo_labels_t, pseudo_labels_a, pseudo_labels_v = [], [], []
            for i in range(len(video_id)):
                if video_id[i] not in train_pseudo_labels:
                    train_pseudo_labels[video_id[i]] = {'T': None, 'A': None, 'V': None}
                    train_pseudo_labels[video_id[i]]['T'] = one_hot_labels_m[i].float()
                    train_pseudo_labels[video_id[i]]['A'] = one_hot_labels_m[i].float()
                    train_pseudo_labels[video_id[i]]['V'] = one_hot_labels_m[i].float()
                pseudo_labels_t.append(train_pseudo_labels[video_id[i]]['T'].unsqueeze(0))
                pseudo_labels_a.append(train_pseudo_labels[video_id[i]]['A'].unsqueeze(0))
                pseudo_labels_v.append(train_pseudo_labels[video_id[i]]['V'].unsqueeze(0))
            pseudo_labels_t = torch.cat(pseudo_labels_t)
            pseudo_labels_a = torch.cat(pseudo_labels_a)
            pseudo_labels_v = torch.cat(pseudo_labels_v)
            labels_t = torch.argmax(pseudo_labels_t, dim=-1, keepdim=True)
            labels_a = torch.argmax(pseudo_labels_a, dim=-1, keepdim=True)
            labels_v = torch.argmax(pseudo_labels_v, dim=-1, keepdim=True)

            t_outputs = t_model(text)
            a_outputs = a_model(audio)
            v_outputs = v_model(vision)

            one_hot_labels_t = F.one_hot(labels_t.view(-1).long(), num_classes=3)
            one_hot_labels_a = F.one_hot(labels_a.view(-1).long(), num_classes=3)
            one_hot_labels_v = F.one_hot(labels_v.view(-1).long(), num_classes=3)
            temp_pseudo_labels_t = generate_unified_pseudo_labels(t_outputs['pred'].detach(), one_hot_labels_t)
            temp_pseudo_labels_a = generate_unified_pseudo_labels(a_outputs['pred'].detach(), one_hot_labels_a)
            temp_pseudo_labels_v = generate_unified_pseudo_labels(v_outputs['pred'].detach(), one_hot_labels_v)

            loss_t = task_criterion(t_outputs['pred'], labels_t.view(-1).long())
            loss_a = task_criterion(a_outputs['pred'], labels_a.view(-1).long())
            loss_v = task_criterion(v_outputs['pred'], labels_v.view(-1).long())

            # update pseudo labels
            pseudo_labels_t = t_momentum * pseudo_labels_t + (1 - t_momentum) * temp_pseudo_labels_t
            pseudo_labels_a = a_momentum * pseudo_labels_a + (1 - a_momentum) * temp_pseudo_labels_a
            pseudo_labels_v = v_momentum * pseudo_labels_v + (1 - v_momentum) * temp_pseudo_labels_v
            for i in range(len(video_id)):
                train_pseudo_labels[video_id[i]]['T'] = pseudo_labels_t[i]
                train_pseudo_labels[video_id[i]]['A'] = pseudo_labels_a[i]
                train_pseudo_labels[video_id[i]]['V'] = pseudo_labels_v[i]

            loss_t.backward(retain_graph=True)
            loss_a.backward(retain_graph=True)
            loss_v.backward(retain_graph=True)

            h_tav = torch.cat((t_outputs['h_l'], a_outputs['h_a'], v_outputs['h_v']), dim=0)
            labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
            if 'SIMS' in hyp_params.dataset:
                labels_tav = (labels_tav > 1).float()
            else:
                labels_tav = (labels_tav >= 1).float()
            loss_c = contrastive_criterion(h_tav, labels_tav)
            loss_c.backward()

            t_optimizer.step()
            a_optimizer.step()
            v_optimizer.step()

            # update ema models
            with torch.no_grad():
                for param, ema_param in zip(t_model.parameters(), ema_t_model.parameters()):
                    ema_param.data = ema_t_model_momentum * ema_param.data + (1 - ema_t_model_momentum) * param.data
                for param, ema_param in zip(a_model.parameters(), ema_a_model.parameters()):
                    ema_param.data = ema_a_model_momentum * ema_param.data + (1 - ema_a_model_momentum) * param.data
                for param, ema_param in zip(v_model.parameters(), ema_v_model.parameters()):
                    ema_param.data = ema_v_model_momentum * ema_param.data + (1 - ema_v_model_momentum) * param.data

            # Collect the results into dictionary
            results['T'].append(t_outputs['pred'])
            results['A'].append(a_outputs['pred'])
            results['V'].append(v_outputs['pred'])
            truths['T'].append(labels_tt)
            truths['A'].append(labels_aa)
            truths['V'].append(labels_vv)

            t_epoch_loss += loss_t.item() * batch_size
            a_epoch_loss += loss_a.item() * batch_size
            v_epoch_loss += loss_v.item() * batch_size
            c_epoch_loss += loss_c.item() * batch_size

        t_epoch_loss /= hyp_params.n_train
        a_epoch_loss /= hyp_params.n_train
        v_epoch_loss /= hyp_params.n_train
        c_epoch_loss /= hyp_params.n_train
        results['T'] = torch.cat(results['T'])
        results['A'] = torch.cat(results['A'])
        results['V'] = torch.cat(results['V'])
        truths['T'] = torch.cat(truths['T'])
        truths['A'] = torch.cat(truths['A'])
        truths['V'] = torch.cat(truths['V'])

        return (t_epoch_loss, a_epoch_loss, v_epoch_loss, c_epoch_loss), results, truths

    def evaluate(models, ema_models, task_criterion, contrastive_criterion, epoch, test=False):
        t_model, a_model, v_model = models
        ema_t_model, ema_a_model, ema_v_model = ema_models
        t_model.eval()
        a_model.eval()
        v_model.eval()
        loader = dataloaders['valid'] if not test else dataloaders['test']

        t_epoch_loss = 0
        a_epoch_loss = 0
        v_epoch_loss = 0
        c_epoch_loss = 0
        results = {
            'T': [],
            'A': [],
            'V': []
        }
        truths = {
            'T': [],
            'A': [],
            'V': []
        }

        with torch.no_grad():
            for batch_step, batch_data in enumerate(loader):
                batch_step += (epoch - 1) * len(loader) + 1
                text = batch_data['text'].to(device)
                audio = batch_data['audio'].to(device)
                vision = batch_data['vision'].to(device)
                video_id = batch_data['id']
                labels_mm = batch_data['labels']['M'].to(device)
                labels_tt = batch_data['labels']['T'].to(device)
                labels_aa = batch_data['labels']['A'].to(device)
                labels_vv = batch_data['labels']['V'].to(device)
                one_hot_labels_m = F.one_hot(labels_mm.view(-1).long(), num_classes=3)

                batch_size = text.size(0)

                pseudo_labels = valid_pseudo_labels if not test else test_pseudo_labels
                pseudo_labels_t, pseudo_labels_a, pseudo_labels_v = [], [], []
                for i in range(len(video_id)):
                    if video_id[i] not in pseudo_labels:
                        pseudo_labels[video_id[i]] = {'T': None, 'A': None, 'V': None}
                        pseudo_labels[video_id[i]]['T'] = one_hot_labels_m[i].float()
                        pseudo_labels[video_id[i]]['A'] = one_hot_labels_m[i].float()
                        pseudo_labels[video_id[i]]['V'] = one_hot_labels_m[i].float()
                    pseudo_labels_t.append(pseudo_labels[video_id[i]]['T'].unsqueeze(0))
                    pseudo_labels_a.append(pseudo_labels[video_id[i]]['A'].unsqueeze(0))
                    pseudo_labels_v.append(pseudo_labels[video_id[i]]['V'].unsqueeze(0))
                pseudo_labels_t = torch.cat(pseudo_labels_t)
                pseudo_labels_a = torch.cat(pseudo_labels_a)
                pseudo_labels_v = torch.cat(pseudo_labels_v)
                labels_t = torch.argmax(pseudo_labels_t, dim=-1, keepdim=True)
                labels_a = torch.argmax(pseudo_labels_a, dim=-1, keepdim=True)
                labels_v = torch.argmax(pseudo_labels_v, dim=-1, keepdim=True)

                t_outputs = t_model(text)
                a_outputs = a_model(audio)
                v_outputs = v_model(vision)
                ema_t_outputs = ema_t_model(text)
                ema_a_outputs = ema_a_model(audio)
                ema_v_outputs = ema_v_model(vision)

                one_hot_labels_t = F.one_hot(labels_t.view(-1).long(), num_classes=3)
                one_hot_labels_a = F.one_hot(labels_a.view(-1).long(), num_classes=3)
                one_hot_labels_v = F.one_hot(labels_v.view(-1).long(), num_classes=3)
                temp_pseudo_labels_t = generate_unified_pseudo_labels(t_outputs['pred'], one_hot_labels_t)
                temp_pseudo_labels_a = generate_unified_pseudo_labels(a_outputs['pred'], one_hot_labels_a)
                temp_pseudo_labels_v = generate_unified_pseudo_labels(v_outputs['pred'], one_hot_labels_v)

                loss_t = task_criterion(ema_t_outputs['pred'], labels_t.view(-1).long())
                loss_a = task_criterion(ema_a_outputs['pred'], labels_a.view(-1).long())
                loss_v = task_criterion(ema_v_outputs['pred'], labels_v.view(-1).long())

                h_tav = torch.cat((ema_t_outputs['h_l'], ema_a_outputs['h_a'], ema_v_outputs['h_v']), dim=0)
                labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
                if 'SIMS' in hyp_params.dataset:
                    labels_tav = (labels_tav > 1).float()
                else:
                    labels_tav = (labels_tav >= 1).float()
                loss_c = contrastive_criterion(h_tav, labels_tav)

                # update pseudo labels
                pseudo_labels_t = t_momentum * pseudo_labels_t + (1 - t_momentum) * temp_pseudo_labels_t
                pseudo_labels_a = a_momentum * pseudo_labels_a + (1 - a_momentum) * temp_pseudo_labels_a
                pseudo_labels_v = v_momentum * pseudo_labels_v + (1 - v_momentum) * temp_pseudo_labels_v
                for i in range(len(video_id)):
                    pseudo_labels[video_id[i]]['T'] = pseudo_labels_t[i]
                    pseudo_labels[video_id[i]]['A'] = pseudo_labels_a[i]
                    pseudo_labels[video_id[i]]['V'] = pseudo_labels_v[i]

                # Collect the results into dictionary
                results['T'].append(ema_t_outputs['pred'])
                results['A'].append(ema_a_outputs['pred'])
                results['V'].append(ema_v_outputs['pred'])
                truths['T'].append(labels_tt)
                truths['A'].append(labels_aa)
                truths['V'].append(labels_vv)

                t_epoch_loss += loss_t.item() * batch_size
                a_epoch_loss += loss_a.item() * batch_size
                v_epoch_loss += loss_v.item() * batch_size
                c_epoch_loss += loss_c.item() * batch_size

        t_epoch_loss /= hyp_params.n_valid if test is False else hyp_params.n_test
        a_epoch_loss /= hyp_params.n_valid if test is False else hyp_params.n_test
        v_epoch_loss /= hyp_params.n_valid if test is False else hyp_params.n_test
        c_epoch_loss /= hyp_params.n_valid if test is False else hyp_params.n_test
        results['T'] = torch.cat(results['T'])
        results['A'] = torch.cat(results['A'])
        results['V'] = torch.cat(results['V'])
        truths['T'] = torch.cat(truths['T'])
        truths['A'] = torch.cat(truths['A'])
        truths['V'] = torch.cat(truths['V'])

        return (t_epoch_loss, a_epoch_loss, v_epoch_loss, c_epoch_loss), results, truths

    text_parameters = sum([param.nelement() for param in t_model.parameters()])
    audio_parameters = sum([param.nelement() for param in a_model.parameters()])
    vision_parameters = sum([param.nelement() for param in v_model.parameters()])
    total_parameters = text_parameters + audio_parameters + vision_parameters
    bert_parameters = sum([param.nelement() for param in t_model.text_model.parameters()])
    print(f'Total Trainable Parameters: {total_parameters}...')
    print(f'BERT Parameters: {bert_parameters}...')
    print(f'TextModel Parameters: {text_parameters - bert_parameters}...')
    print(f'AudioModel Parameters: {audio_parameters}...')
    print(f'VisionModel Parameters: {vision_parameters}...')
    inf = 1e8
    valid_best_loss = {
        'T': inf,
        'A': inf,
        'V': inf
    }
    test_best_acc = {
        'T': 0,
        'A': 0,
        'V': 0
    }
    curr_patience = hyp_params.patience
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()

        t_momentum = get_momentum(init_t_momentum, epoch, gamma=hyp_params.t_momentum_gamma)
        a_momentum = get_momentum(init_a_momentum, epoch, gamma=hyp_params.a_momentum_gamma)
        v_momentum = get_momentum(init_v_momentum, epoch, gamma=hyp_params.v_momentum_gamma)
        ema_t_model_momentum = get_momentum(init_ema_t_model_momentum, epoch, gamma=hyp_params.ema_t_model_momentum_gamma)
        ema_a_model_momentum = get_momentum(init_ema_a_model_momentum, epoch, gamma=hyp_params.ema_a_model_momentum_gamma)
        ema_v_model_momentum = get_momentum(init_ema_v_model_momentum, epoch, gamma=hyp_params.ema_v_model_momentum_gamma)

        train_losses, train_results, train_truths = train((t_model, a_model, v_model),
                                                          (ema_t_model, ema_a_model, ema_v_model),
                                                          (t_optimizer, a_optimizer, v_optimizer),
                                                          task_criterion, contrastive_criterion, epoch)
        valid_losses, _, _ = evaluate((t_model, a_model, v_model),
                                      (ema_t_model, ema_a_model, ema_v_model),
                                      task_criterion, contrastive_criterion, epoch, test=False)
        test_losses, results, truths = evaluate((t_model, a_model, v_model),
                                                (ema_t_model, ema_a_model, ema_v_model),
                                                task_criterion, contrastive_criterion, epoch, test=True)

        end = time.time()
        duration = end - start

        t_train_loss, a_train_loss, v_train_loss, c_train_loss = train_losses
        t_valid_loss, a_valid_loss, v_valid_loss, c_valid_loss = valid_losses
        t_test_loss, a_test_loss, v_test_loss, c_test_loss = test_losses
        print("-" * 50)
        print('Epoch {:2d} | Time {:5.4f} sec'.format(epoch, duration))
        print(
            'Train Text Loss {:5.4f} | Train Audio Loss {:5.4f} | Train Vision Loss {:5.4f} | Train Contrastive Loss {:5.4f}'.format(
                t_train_loss, a_train_loss, v_train_loss, c_train_loss))
        print(
            'Valid Text Loss {:5.4f} | Valid Audio Loss {:5.4f} | Valid Vision Loss {:5.4f} | Valid Contrastive Loss {:5.4f}'.format(
                t_valid_loss, a_valid_loss, v_valid_loss, c_valid_loss))
        print(
            'Test  Text Loss {:5.4f} | Test  Audio Loss {:5.4f} | Test  Vision Loss {:5.4f} | Test  Contrastive Loss {:5.4f}'.format(
                t_test_loss, a_test_loss, v_test_loss, c_test_loss))

        if 'SIMS' in hyp_params.dataset:
            t_ans = eval_sims_classification(train_results['T'], train_truths['T'])['Has0_acc_2']
            a_ans = eval_sims_classification(train_results['A'], train_truths['A'])['Has0_acc_2']
            v_ans = eval_sims_classification(train_results['V'], train_truths['V'])['Has0_acc_2']
        else:
            t_ans = eval_mosi_classification(train_results['T'], train_truths['T'])['Has0_acc_2']
            a_ans = eval_mosi_classification(train_results['A'], train_truths['A'])['Has0_acc_2']
            v_ans = eval_mosi_classification(train_results['V'], train_truths['V'])['Has0_acc_2']
        train_res = {'T': t_ans, 'A': a_ans, 'V': v_ans}
        print('Current Train Results:')
        for idx, key in enumerate(train_res):
            if idx == len(train_res) - 1:
                print(f'{key}: {train_res[key]:.4f}')
            else:
                print(f'{key}: {train_res[key]:.4f}', end=' | ')

        if 'SIMS' in hyp_params.dataset:
            t_ans = eval_sims_classification(results['T'], truths['T'])['Has0_acc_2']
            a_ans = eval_sims_classification(results['A'], truths['A'])['Has0_acc_2']
            v_ans = eval_sims_classification(results['V'], truths['V'])['Has0_acc_2']
        else:
            t_ans = eval_mosi_classification(results['T'], truths['T'])['Has0_acc_2']
            a_ans = eval_mosi_classification(results['A'], truths['A'])['Has0_acc_2']
            v_ans = eval_mosi_classification(results['V'], truths['V'])['Has0_acc_2']
        test_res = {'T': t_ans, 'A': a_ans, 'V': v_ans}
        print('Current Test Results:')
        for idx, key in enumerate(test_res):
            if idx == len(test_res) - 1:
                print(f'{key}: {test_res[key]:.4f}')
            else:
                print(f'{key}: {test_res[key]:.4f}', end=' | ')

        t_valid_loss = round(t_valid_loss, 4)
        a_valid_loss = round(a_valid_loss, 4)
        v_valid_loss = round(v_valid_loss, 4)
        is_better = False
        if t_valid_loss < valid_best_loss['T']:
            is_better = True
            valid_best_loss['T'] = t_valid_loss
            test_best_acc['T'] = t_ans
            save_model(hyp_params, t_model, names={'model_name': 'Gen_Pseudo_TextModel'})
            save_model(hyp_params, ema_t_model, names={'model_name': 'Pseudo_TextModel'})
        if a_valid_loss < valid_best_loss['A']:
            is_better = True
            valid_best_loss['A'] = a_valid_loss
            test_best_acc['A'] = a_ans
            save_model(hyp_params, a_model, names={'model_name': 'Gen_Pseudo_AudioModel'})
            save_model(hyp_params, ema_a_model, names={'model_name': 'Pseudo_AudioModel'})
        if v_valid_loss < valid_best_loss['V']:
            is_better = True
            valid_best_loss['V'] = v_valid_loss
            test_best_acc['V'] = v_ans
            save_model(hyp_params, v_model, names={'model_name': 'Gen_Pseudo_VisionModel'})
            save_model(hyp_params, ema_v_model, names={'model_name': 'Pseudo_VisionModel'})
        if is_better:
            curr_patience = hyp_params.patience
            with open(f'{hyp_params.dataset}_train_pseudo_labels.pkl', 'wb') as f:
                pickle.dump(train_pseudo_labels, f)
            with open(f'{hyp_params.dataset}_valid_pseudo_labels.pkl', 'wb') as f:
                pickle.dump(valid_pseudo_labels, f)
            with open(f'{hyp_params.dataset}_test_pseudo_labels.pkl', 'wb') as f:
                pickle.dump(test_pseudo_labels, f)
        else:
            curr_patience -= 1
        print('Current Test Best Results:')
        for idx, key in enumerate(test_best_acc):
            if idx == len(test_best_acc) - 1:
                print(f'{key}: {test_best_acc[key]:.4f}')
            else:
                print(f'{key}: {test_best_acc[key]:.4f}', end=' | ')
        if curr_patience <= 0:
            break

    eval_metric = hyp_params.eval_metric
    best_t_model = load_model(hyp_params, names={'model_name': 'Gen_Pseudo_TextModel'})
    best_a_model = load_model(hyp_params, names={'model_name': 'Gen_Pseudo_AudioModel'})
    best_v_model = load_model(hyp_params, names={'model_name': 'Gen_Pseudo_VisionModel'})
    best_ema_t_model = load_model(hyp_params, names={'model_name': 'Pseudo_TextModel'})
    best_ema_a_model = load_model(hyp_params, names={'model_name': 'Pseudo_AudioModel'})
    best_ema_v_model = load_model(hyp_params, names={'model_name': 'Pseudo_VisionModel'})
    ans = {}
    epoch = -1
    _, results, truths = evaluate((best_t_model, best_a_model, best_v_model),
                                  (best_ema_t_model, best_ema_a_model, best_ema_v_model),
                                  task_criterion, contrastive_criterion, epoch, test=True)
    if 'SIMS' in hyp_params.dataset:
        ans['T'] = eval_sims_classification(results['T'], truths['T'])[eval_metric]
        ans['A'] = eval_sims_classification(results['A'], truths['A'])[eval_metric]
        ans['V'] = eval_sims_classification(results['V'], truths['V'])[eval_metric]
    else:
        ans['T'] = eval_mosi_classification(results['T'], truths['T'])[eval_metric]
        ans['A'] = eval_mosi_classification(results['A'], truths['A'])[eval_metric]
        ans['V'] = eval_mosi_classification(results['V'], truths['V'])[eval_metric]

    return ans


def compute_alpha(logits, base_labels, reversed_labels):
    one_hot_base_labels = base_labels
    one_hot_reversed_labels = reversed_labels
    probs = F.softmax(logits, dim=-1)
    g_base = torch.norm(probs - one_hot_base_labels, p=2, dim=-1)
    g_reverse = torch.norm(probs - one_hot_reversed_labels, p=2, dim=-1)
    alpha = (g_base - g_reverse) / (g_base + g_reverse)
    return alpha


def reverse_labels(logits, one_hot_labels):
    batch_size, num_classes = one_hot_labels.shape
    reversed_labels = torch.zeros_like(one_hot_labels)
    probs = F.softmax(logits, dim=-1)
    for i in range(batch_size):
        original_class = torch.argmax(one_hot_labels[i]).item()
        min_norm = float('inf')
        best_class = original_class
        for candidate in range(num_classes):
            candidate_label = torch.eye(num_classes)[candidate].to(logits.device)
            g_candidate = (probs[i] - candidate_label).detach()
            norm = torch.norm(g_candidate, p=2)
            if norm < min_norm:
                min_norm = norm
                best_class = candidate
        reversed_labels[i] = torch.eye(num_classes)[best_class]
    return reversed_labels


def generate_unified_pseudo_labels(logits, one_hot_labels):
    base_labels = one_hot_labels
    reversed_labels = reverse_labels(logits, one_hot_labels)
    alpha = compute_alpha(logits, base_labels, reversed_labels)
    pseudo_labels = (1 - alpha.unsqueeze(1)) * base_labels + alpha.unsqueeze(1) * reversed_labels
    return pseudo_labels

def get_momentum(init_momentum, epoch, gamma):
    return 1 - (1 - init_momentum) / (epoch ** gamma)
