import pickle
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

def initiate(hyp_params, dataloaders, pretrained_models, device):
    model = models.CMCModel(hyp_params)
    if pretrained_models is not None:
        model = transfer_models(model, pretrained_models)

    if hyp_params.use_cuda:
        model = model.to(device)

    bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_params = list(model.text_model.named_parameters())
    bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
    bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
    model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
    optimizer_grouped_parameters = [
        {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
        {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
        {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters)
    task_criterion = getattr(nn, hyp_params.criterion)()
    contrastive_criterion = models.SupConLoss(temperature=hyp_params.pretrain_temperature)
    settings = {'model': model,
                'optimizer': optimizer,
                'task_criterion': task_criterion,
                'contrastive_criterion': contrastive_criterion}
    return train_model(settings, hyp_params, dataloaders, device)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, dataloaders, device):
    model = settings['model']
    optimizer = settings['optimizer']
    task_criterion = settings['task_criterion']
    contrastive_criterion = settings['contrastive_criterion']
    if hyp_params.is_pseudo:
        with open(f'{hyp_params.dataset}_train_pseudo_labels.pkl', 'rb') as f:
            train_pseudo_labels = pickle.load(f)
        with open(f'{hyp_params.dataset}_valid_pseudo_labels.pkl', 'rb') as f:
            valid_pseudo_labels = pickle.load(f)
        with open(f'{hyp_params.dataset}_test_pseudo_labels.pkl', 'rb') as f:
            test_pseudo_labels = pickle.load(f)
    def train(model, optimizer, task_criterion, contrastive_criterion):
        epoch_loss = 0
        model.train()

        for batch_data in dataloaders['train']:
            text = batch_data['text'].to(device)
            audio = batch_data['audio'].to(device)
            vision = batch_data['vision'].to(device)
            labels = batch_data['labels']['M'].to(device)
            if not hyp_params.is_pseudo:
                labels_t = batch_data['labels']['T'].to(device)
                labels_a = batch_data['labels']['A'].to(device)
                labels_v = batch_data['labels']['V'].to(device)
            else:
                video_id = batch_data['id']
                pseudo_labels_t, pseudo_labels_a, pseudo_labels_v = [], [], []
                for i in range(len(video_id)):
                    pseudo_labels_t.append(train_pseudo_labels[video_id[i]]['T'].unsqueeze(0))
                    pseudo_labels_a.append(train_pseudo_labels[video_id[i]]['A'].unsqueeze(0))
                    pseudo_labels_v.append(train_pseudo_labels[video_id[i]]['V'].unsqueeze(0))
                pseudo_labels_t = torch.cat(pseudo_labels_t)
                pseudo_labels_a = torch.cat(pseudo_labels_a)
                pseudo_labels_v = torch.cat(pseudo_labels_v)
                labels_t = torch.argmax(pseudo_labels_t, dim=-1, keepdim=True)
                labels_a = torch.argmax(pseudo_labels_a, dim=-1, keepdim=True)
                labels_v = torch.argmax(pseudo_labels_v, dim=-1, keepdim=True)

            batch_size = text.size(0)
            model.zero_grad()

            outputs = model(text, audio, vision)
            loss_m = task_criterion(outputs['pred'], labels.view(-1).long())
            loss_t = task_criterion(outputs['pred_t'], labels_t.view(-1).long())
            loss_a = task_criterion(outputs['pred_a'], labels_a.view(-1).long())
            loss_v = task_criterion(outputs['pred_v'], labels_v.view(-1).long())
            loss_task = loss_m + loss_t + loss_a + loss_v

            h_tav = torch.cat((outputs['h_l'], outputs['h_a'], outputs['h_v']), dim=0)
            labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
            if 'SIMS' in hyp_params.dataset:
                labels_tav = (labels_tav > 1).float()
            else:
                labels_tav = (labels_tav >= 1).float()
            loss_c = contrastive_criterion(h_tav, labels_tav)

            loss = loss_task + loss_c
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size

        epoch_loss /= hyp_params.n_train

        return epoch_loss

    def evaluate(model, task_criterion, contrastive_criterion, test=False, test_mode=None):
        model.eval()
        if not test:
            loader = dataloaders['valid']
        else:
            loader = dataloaders[test_mode]

        epoch_loss = 0
        results = []
        truths = []

        with torch.no_grad():
            for batch_data in loader:
                text = batch_data['text'].to(device)
                audio = batch_data['audio'].to(device)
                vision = batch_data['vision'].to(device)
                labels = batch_data['labels']['M'].to(device)
                if not hyp_params.is_pseudo:
                    labels_t = batch_data['labels']['T'].to(device)
                    labels_a = batch_data['labels']['A'].to(device)
                    labels_v = batch_data['labels']['V'].to(device)
                else:
                    video_id = batch_data['id']
                    pseudo_labels = valid_pseudo_labels if not test else test_pseudo_labels
                    pseudo_labels_t, pseudo_labels_a, pseudo_labels_v = [], [], []
                    for i in range(len(video_id)):
                        pseudo_labels_t.append(pseudo_labels[video_id[i]]['T'].unsqueeze(0))
                        pseudo_labels_a.append(pseudo_labels[video_id[i]]['A'].unsqueeze(0))
                        pseudo_labels_v.append(pseudo_labels[video_id[i]]['V'].unsqueeze(0))
                    pseudo_labels_t = torch.cat(pseudo_labels_t)
                    pseudo_labels_a = torch.cat(pseudo_labels_a)
                    pseudo_labels_v = torch.cat(pseudo_labels_v)
                    labels_t = torch.argmax(pseudo_labels_t, dim=-1, keepdim=True)
                    labels_a = torch.argmax(pseudo_labels_a, dim=-1, keepdim=True)
                    labels_v = torch.argmax(pseudo_labels_v, dim=-1, keepdim=True)

                batch_size = text.size(0)

                outputs = model(text, audio, vision)
                loss_m = task_criterion(outputs['pred'], labels.view(-1).long())
                loss_t = task_criterion(outputs['pred_t'], labels_t.view(-1).long())
                loss_a = task_criterion(outputs['pred_a'], labels_a.view(-1).long())
                loss_v = task_criterion(outputs['pred_v'], labels_v.view(-1).long())
                loss_task = loss_m + loss_t + loss_a + loss_v

                h_tav = torch.cat((outputs['h_l'], outputs['h_a'], outputs['h_v']), dim=0)
                labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
                if 'SIMS' in hyp_params.dataset:
                    labels_tav = (labels_tav > 1).float()
                else:
                    labels_tav = (labels_tav >= 1).float()
                loss_c = contrastive_criterion(h_tav, labels_tav)

                loss = loss_task + loss_c

                # Collect the results into dictionary
                results.append(outputs['pred'])
                truths.append(labels)

                epoch_loss += loss.item() * batch_size

        epoch_loss /= hyp_params.n_valid if test is False else hyp_params.n_test
        results = torch.cat(results)
        truths = torch.cat(truths)

        return epoch_loss, results, truths

    total_parameters = sum([param.nelement() for param in model.parameters()])
    bert_parameters = sum([param.nelement() for param in model.text_model.parameters()])
    print(f'Total Trainable Parameters: {total_parameters}...')
    print(f'BERT Parameters: {bert_parameters}...')
    print(f'CMCModel Parameters: {total_parameters - bert_parameters}...')
    valid_best_loss = 1e8
    test_best_value = {
        'Has0_acc_2': 0,
        'Has0_F1_score': 0,
        'Non0_acc_2': 0,
        'Non0_F1_score': 0,
        'Acc_3': 0,
        'F1_score_3': 0
    }
    curr_patience = hyp_params.patience
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()

        train_loss = train(model, optimizer, task_criterion, contrastive_criterion)
        valid_loss, _, _ = evaluate(model, task_criterion, contrastive_criterion, test=False)
        _, results, truths = evaluate(model, task_criterion, contrastive_criterion, test=True, test_mode='test')

        end = time.time()
        duration = end - start

        print("-" * 50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f}'.format(
            epoch, duration, train_loss, valid_loss))

        if 'SIMS' in hyp_params.dataset:
            ans = eval_sims_classification(results, truths)
        else:
            ans = eval_mosi_classification(results, truths)
        is_better = False
        if valid_loss < valid_best_loss:
            is_better = True
            for key in ans:
                test_best_value[key] = ans[key]
                save_model(hyp_params, model, names={'model_name': f'CMCModel_{key}'})
        if is_better:
            curr_patience = hyp_params.patience
        else:
            curr_patience -= 1
        print(f'Current Test Best Results:')
        for idx, key in enumerate(test_best_value):
            if idx == len(test_best_value) - 1:
                print(f'{key}: {test_best_value[key]:.4f}')
            else:
                print(f'{key}: {test_best_value[key]:.4f}', end=' | ')
        if curr_patience <= 0:
            break

    eval_metric = hyp_params.eval_metric
    best_model = load_model(hyp_params, names={'model_name': f'CMCModel_{eval_metric}'})

    ans = {}
    if 'SIMS' in hyp_params.dataset:
        print('D_test...')
        _, results, truths = evaluate(best_model, task_criterion, contrastive_criterion, test=True, test_mode='test')
        ans['D_test'] = eval_sims_classification(results, truths)[eval_metric]
        print('D_msc...')
        _, results, truths = evaluate(best_model, task_criterion, contrastive_criterion, test=True, test_mode='D_msc')
        ans['D_msc'] = eval_sims_classification(results, truths)[eval_metric]
        print('D_msi...')
        _, results, truths = evaluate(best_model, task_criterion, contrastive_criterion, test=True, test_mode='D_msi')
        ans['D_msi'] = eval_sims_classification(results, truths)[eval_metric]
    else:
        print('D_test...')
        _, results, truths = evaluate(best_model, task_criterion, contrastive_criterion, test=True, test_mode='test')
        ans['D_test'] = eval_mosi_classification(results, truths)[eval_metric]

    return ans
