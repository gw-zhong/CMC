import csv
import os
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description='Robust Multimodal Emotion Recognition')
parser.add_argument('-f', default='', type=str)

# Tasks
parser.add_argument('--dataset', type=str, default='SIMS-v2',
                    help='dataset to use (SIMS / SIMS-v2 / MOSI / MOSEI)')
parser.add_argument('--is_pseudo', action='store_true',
                    help='whether to use the pseudo labels (default: false)')

args = parser.parse_args()


if __name__ == '__main__':
    param_grid = {
        'transformer_layers': [1, 2, 3, 4, 5],
        'nhead': [1, 2, 4, 8],
        'out_dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    grid = ParameterGrid(param_grid)
    prefix = 'is_pseudo_' if args.is_pseudo else ''
    if 'SIMS' in args.dataset:
        with open(f'{prefix}{args.dataset}_tuning_result.csv', 'w+') as out:
            writer = csv.writer(out)
            writer.writerow([
                'transformer_layers',
                'nhead',
                'out_dropout',
                'test',
                'D_msc',
                'D_msi'
            ])
        best_results = {
            'transformer_layers': 0,
            'nhead': 0,
            'out_dropout': 0,
            'test': 0,
            'D_msc': 0,
            'D_msi': 0
        }
    else:
        with open(f'{prefix}{args.dataset}_tuning_result.csv', 'w+') as out:
            writer = csv.writer(out)
            writer.writerow([
                'transformer_layers',
                'nhead',
                'out_dropout',
                'test'
            ])
        best_results = {
            'transformer_layers': 0,
            'nhead': 0,
            'out_dropout': 0,
            'test': 0
        }
    param_num = 0

    all_params = list(grid)
    total = len(all_params)
    next_params = {
        'transformer_layers': 1,
        'nhead': 4,
        'out_dropout': 0.5
    }
    next_index = all_params.index(next_params)
    remaining_params = all_params[next_index:]
    remaining = len(remaining_params)
    completed = next_index
    print(f'A total of {total} sets of parameters need to be evaluated. {completed} sets have been completed, and {remaining} sets remain to be assessed.')
    print(f'Continue the search from parameter combination {next_index + 1}/{total}: {next_params}')
    for params in remaining_params:
        param_num += 1

        args.transformer_layers = params['transformer_layers']
        args.nhead = params['nhead']
        args.out_dropout = params['out_dropout']
        print('=' * 40)
        print('Hyperparameter:' + '{}/{}'.format(param_num, remaining).rjust(25))
        print('-' * 40)
        print('transformer_layers'.ljust(35) + '= ' + str(args.transformer_layers))
        print('nhead'.ljust(35) + '= ' + str(args.nhead))
        print('out_dropout'.ljust(35) + '= ' + str(args.out_dropout))
        print('=' * 40)

        if args.is_pseudo:
            result = subprocess.run(['python', 'main.py',
                                     '--dataset', f'{args.dataset}',
                                     '--transformer_layers', f'{params["transformer_layers"]}',
                                     '--nhead', f'{params["nhead"]}',
                                     '--out_dropout', f'{params["out_dropout"]}',
                                     '--is_pseudo'],
                                    capture_output=True,
                                    text=True,
                                    check=True)
            print('phase1 result:\n', result.stdout)

            result = subprocess.run(['python', 'main.py',
                                     '--dataset', f'{args.dataset}',
                                     '--transformer_layers', f'{params["transformer_layers"]}',
                                     '--nhead', f'{params["nhead"]}',
                                     '--out_dropout', f'{params["out_dropout"]}',
                                     '--is_pseudo',
                                     '--finetune',
                                     '--pretrained_model'],
                                    capture_output=True,
                                    text=True,
                                    check=True)
            print('phase2 result\n:', result.stdout)
        else:
            result = subprocess.run(['python', 'main.py',
                                     '--dataset', f'{args.dataset}',
                                     '--transformer_layers', f'{params["transformer_layers"]}',
                                     '--nhead', f'{params["nhead"]}',
                                     '--out_dropout', f'{params["out_dropout"]}',
                                     '--finetune'],
                                    capture_output=True,
                                    text=True,
                                    check=True)
            print('phase1 result:\n', result.stdout)

            result = subprocess.run(['python', 'main.py',
                                     '--dataset', f'{args.dataset}',
                                     '--transformer_layers', f'{params["transformer_layers"]}',
                                     '--nhead', f'{params["nhead"]}',
                                     '--out_dropout', f'{params["out_dropout"]}',
                                     '--finetune',
                                     '--pretrained_model'],
                                    capture_output=True,
                                    text=True,
                                    check=True)
            print('phase2 result\n:', result.stdout)

        lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        if 'SIMS' in args.dataset:
            target_lines = lines[-3:]
        else:
            target_lines = lines[-1:]

        ans = {}
        for line in target_lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                try:
                    ans[key.strip()] = float(value.strip())
                except ValueError:
                    ans[key.strip()] = value.strip()

        print('-' * 40)
        if 'SIMS' in args.dataset:
            if ans['test'] > best_results['test']:
                best_results['test'] = ans['test']
                best_results['D_msc'] = ans['D_msc']
                best_results['D_msi'] = ans['D_msi']
                best_results['transformer_layers'] = params['transformer_layers']
                best_results['nhead'] = params['nhead']
                best_results['out_dropout'] = params['out_dropout']
                print('Found new best result!')
        else:
            if ans['test'] > best_results['test']:
                best_results['test'] = ans['test']
                best_results['transformer_layers'] = params['transformer_layers']
                best_results['nhead'] = params['nhead']
                best_results['out_dropout'] = params['out_dropout']
                print('Found new best result!')
        print('Current best result is:')
        for key, value in best_results.items():
            print(f'{key}: {value}')
        if 'SIMS' in args.dataset:
            with open(f'{prefix}{args.dataset}_tuning_result.csv', 'a+') as out:
                writer = csv.writer(out)
                writer.writerow([
                    params['transformer_layers'],
                    params['nhead'],
                    params['out_dropout'],
                    ans['test'],
                    ans['D_msc'],
                    ans['D_msi']
                ])
        else:
            with open(f'{prefix}{args.dataset}_tuning_result.csv', 'a+') as out:
                writer = csv.writer(out)
                writer.writerow([
                    params['transformer_layers'],
                    params['nhead'],
                    params['out_dropout'],
                    ans['test']
                ])
    print('-' * 40)
    print('Grid Search Over!')
    print('Best result is:')
    for key, value in best_results.items():
        print(f'{key}: {value}')