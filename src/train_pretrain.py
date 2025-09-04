from torch import nn
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

    if hyp_params.use_cuda:
        t_model = t_model.to(device)
        a_model = a_model.to(device)
        v_model = v_model.to(device)

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
    t_optimizer = settings['t_optimizer']
    a_optimizer = settings['a_optimizer']
    v_optimizer = settings['v_optimizer']
    task_criterion = settings['task_criterion']
    contrastive_criterion = settings['contrastive_criterion']

    def train(models, optimizers, task_criterion, contrastive_criterion):
        t_model, a_model, v_model = models
        t_optimizer, a_optimizer, v_optimizer = optimizers
        t_epoch_loss = 0
        a_epoch_loss = 0
        v_epoch_loss = 0
        c_epoch_loss = 0
        t_model.train()
        a_model.train()
        v_model.train()

        for batch_data in dataloaders['train']:
            text = batch_data['text'].to(device)
            audio = batch_data['audio'].to(device)
            vision = batch_data['vision'].to(device)
            labels_t = batch_data['labels']['T'].to(device)
            labels_a = batch_data['labels']['A'].to(device)
            labels_v = batch_data['labels']['V'].to(device)

            batch_size = text.size(0)
            t_model.zero_grad()
            a_model.zero_grad()
            v_model.zero_grad()

            t_outputs = t_model(text)
            loss_t = task_criterion(t_outputs['pred'], labels_t.view(-1).long())
            loss_t.backward(retain_graph=True)

            a_outputs = a_model(audio)
            loss_a = task_criterion(a_outputs['pred'], labels_a.view(-1).long())
            loss_a.backward(retain_graph=True)

            v_outputs = v_model(vision)
            loss_v = task_criterion(v_outputs['pred'], labels_v.view(-1).long())
            loss_v.backward(retain_graph=True)

            h_tav = torch.cat((t_outputs['h_l'], a_outputs['h_a'], v_outputs['h_v']), dim=0)
            labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
            labels_tav = (labels_tav > 1).float()
            loss_c = contrastive_criterion(h_tav, labels_tav)
            loss_c.backward()

            t_optimizer.step()
            a_optimizer.step()
            v_optimizer.step()

            t_epoch_loss += loss_t.item() * batch_size
            a_epoch_loss += loss_a.item() * batch_size
            v_epoch_loss += loss_v.item() * batch_size
            c_epoch_loss += loss_c.item() * batch_size

        t_epoch_loss /= hyp_params.n_train
        a_epoch_loss /= hyp_params.n_train
        v_epoch_loss /= hyp_params.n_train
        c_epoch_loss /= hyp_params.n_train

        return t_epoch_loss, a_epoch_loss, v_epoch_loss, c_epoch_loss

    def evaluate(models, task_criterion, contrastive_criterion, test=False):
        t_model, a_model, v_model = models
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
            for batch_data in loader:
                text = batch_data['text'].to(device)
                audio = batch_data['audio'].to(device)
                vision = batch_data['vision'].to(device)
                labels_t = batch_data['labels']['T'].to(device)
                labels_a = batch_data['labels']['A'].to(device)
                labels_v = batch_data['labels']['V'].to(device)

                batch_size = text.size(0)
                t_outputs = t_model(text)
                loss_t = task_criterion(t_outputs['pred'], labels_t.view(-1).long())

                a_outputs = a_model(audio)
                loss_a = task_criterion(a_outputs['pred'], labels_a.view(-1).long())

                v_outputs = v_model(vision)
                loss_v = task_criterion(v_outputs['pred'], labels_v.view(-1).long())

                h_tav = torch.cat((t_outputs['h_l'], a_outputs['h_a'], v_outputs['h_v']), dim=0)
                labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
                labels_tav = (labels_tav > 1).float()
                loss_c = contrastive_criterion(h_tav, labels_tav)

                # Collect the results into dictionary
                results['T'].append(t_outputs['pred'])
                results['A'].append(a_outputs['pred'])
                results['V'].append(v_outputs['pred'])
                truths['T'].append(labels_t)
                truths['A'].append(labels_a)
                truths['V'].append(labels_v)

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

        train_losses = train((t_model, a_model, v_model),
                           (t_optimizer, a_optimizer, v_optimizer), task_criterion, contrastive_criterion)
        valid_losses, _, _ = evaluate((t_model, a_model, v_model), task_criterion, contrastive_criterion, test=False)
        _, results, truths = evaluate((t_model, a_model, v_model), task_criterion, contrastive_criterion, test=True)

        end = time.time()
        duration = end - start

        t_train_loss, a_train_loss, v_train_loss, c_train_loss = train_losses
        t_valid_loss, a_valid_loss, v_valid_loss, c_valid_loss = valid_losses
        print("-" * 50)
        print('Epoch {:2d} | Time {:5.4f} sec'.format(
            epoch, duration))
        print('Train Text Loss {:5.4f} | Train Audio Loss {:5.4f} | Train Vision Loss {:5.4f} | Train Contrastive Loss {:5.4f}'.format(
            t_train_loss, a_train_loss, v_train_loss, c_train_loss))
        print('Valid Text Loss {:5.4f} | Valid Audio Loss {:5.4f} | Valid Vision Loss {:5.4f} | Valid Contrastive Loss {:5.4f}'.format(
            t_valid_loss, a_valid_loss, v_valid_loss, c_valid_loss))

        t_ans = eval_sims_classification(results['T'], truths['T'])['Has0_acc_2']
        a_ans = eval_sims_classification(results['A'], truths['A'])['Has0_acc_2']
        v_ans = eval_sims_classification(results['V'], truths['V'])['Has0_acc_2']
        is_better = False
        if t_valid_loss < valid_best_loss['T']:
            is_better = True
            valid_best_loss['T'] = t_valid_loss
            test_best_acc['T'] = t_ans
            save_model(hyp_params, t_model, names={'model_name': 'TextModel'})
        if a_valid_loss < valid_best_loss['A']:
            is_better = True
            valid_best_loss['A'] = a_valid_loss
            test_best_acc['A'] = a_ans
            save_model(hyp_params, a_model, names={'model_name': 'AudioModel'})
        if v_valid_loss < valid_best_loss['V']:
            is_better = True
            valid_best_loss['V'] = v_valid_loss
            test_best_acc['V'] = v_ans
            save_model(hyp_params, v_model, names={'model_name': 'VisionModel'})
        if is_better:
            curr_patience = hyp_params.patience
        else:
            curr_patience -= 1
        print(f'Current Test Best Results:')
        for idx, key in enumerate(test_best_acc):
            if idx == len(test_best_acc) - 1:
                print(f'{key}: {test_best_acc[key]:.4f}')
            else:
                print(f'{key}: {test_best_acc[key]:.4f}', end=' | ')
        if curr_patience <= 0:
            break

    eval_metric = hyp_params.eval_metric
    best_t_model = load_model(hyp_params, names={'model_name': 'TextModel'})
    best_a_model = load_model(hyp_params, names={'model_name': 'AudioModel'})
    best_v_model = load_model(hyp_params, names={'model_name': 'VisionModel'})
    ans = {}
    _, results, truths = evaluate((best_t_model, best_a_model, best_v_model),
                                  task_criterion, contrastive_criterion, test=True)
    ans['T'] = eval_sims_classification(results['T'], truths['T'])[eval_metric]
    ans['A'] = eval_sims_classification(results['A'], truths['A'])[eval_metric]
    ans['V'] = eval_sims_classification(results['V'], truths['V'])[eval_metric]

    return ans
