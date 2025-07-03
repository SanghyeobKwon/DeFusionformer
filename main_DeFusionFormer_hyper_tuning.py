import torch
from data_provider.data_loader import *
import itertools

import argparse
from exp.exp_main_Multi import Exp_Main_Multi
import random

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

import json
from functools import reduce



def Graidsearch(paramgrid, Exp, args, iter_max):
    best_loss = float('inf')
    best_params = None
    i = 1
    for combination in itertools.product(*paramgrid.values()):
        setting = dict(zip(paramgrid.keys(), combination))
        for key, value in setting.items():
            setattr(args, key, value)

        exp = Exp(args)
        recording = (f'{args.model_id}_')
        exp.train(recording)
        vali_loss = exp.vali_loss

        if vali_loss < best_loss:
            best_loss = vali_loss
            best_params = setting

        if i >= iter_max:
            break
        i += 1

    return best_params, best_loss

def Randomsearch(paramgrid, Exp, args, iter_max):
    best_loss = float('inf')
    best_params = None

    for _ in range(iter_max):
        random_params = {
            key: random.randint(min(values['values']), max(values['values'])) if values['type'] == int else round(random.uniform(min(values['values']), max(values['values'])), 10)
            for key, values in paramgrid.items()
        }

        for key, value in random_params.items():
            if key == 'd_model':
                value = value if value % 2 == 0 else value + 1
            setattr(args, key, value)
            random_params[key] = value

        exp = Exp(args)
        recording = (f'{args.model_id}_')
        exp.train(recording)
        vali_loss = exp.vali_loss

        if vali_loss < best_loss:
            best_loss = vali_loss
            best_params = random_params

    return best_params, best_loss


def BayesianOptimization(paramgrid, Exp, args, iter_max):
    search_space = []
    for key, values in paramgrid.items():
        param_type = values['type']  # 데이터 타입
        param_values = values['values']  # 값 리스트

        if param_type == int:
            search_space.append(Integer(min(param_values), max(param_values), name=key))
        elif param_type == float:
            search_space.append(Real(min(param_values), max(param_values), name=key))
        else:
            search_space.append(Categorical(param_values, name=key))


    def objective_function(params):

        param_dict = {key: value for key, value in zip(paramgrid.keys(), params)}


        for key, value in param_dict.items():
            if key == 'd_model':
                value = value if value % 2 == 0 else value + 1
            setattr(args, key, value)
            param_dict[key] = value

        # 실험 실행
        print(f"Running with params: {param_dict}")
        exp = Exp(args)
        recording = f'{args.model_id}_'
        exp.train(recording)
        vali_loss = exp.vali_loss

        return vali_loss

    result = gp_minimize(
        func=objective_function,
        dimensions=search_space,
        n_calls=iter_max,
        random_state=42,
        n_initial_points=3
    )

    best_params = {key: value for key, value in zip(paramgrid.keys(), result.x)}
    best_loss = result.fun

    return best_params, best_loss



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        return super().default(obj)


def main(seed, num, target, root,model='Transformer', types = 'test', search_method = 'gridsearch'):
    fix_seed = seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='DeFusionformer')
    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')

    parser.add_argument('--model_id', type=str, default=f'test_{num}', help='model id')
    parser.add_argument('--model', type=str, default=f'{model}',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default=f'Multi_Input', help='dataset type')
    parser.add_argument('--root_path', type=str, default=root, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=f'{num}.csv', help='data file')

    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default=f'{target}', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


    # forecasting task
    parser.add_argument('--seq_len_L', type=int, default=96, help='mid_input sequence length')
    parser.add_argument('--seq_len_S', type=int, default=12, help='short_input sequence length')
    parser.add_argument('--label_len', type=int, default=24, help='start token length')

    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

    # model define
    parser.add_argument('--moving_avg_L', type=int, default=12, help='window size of moving average')
    parser.add_argument('--moving_avg_S', type=int, default=9, help='window size of moving average')

    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=150, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    #하이퍼파라미터 튜닝
    parser.add_argument('--search_method', type=str, default=f'{search_method}', help='[GridSearch, RandomSearch, BayesianSearch]')


    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)


    Exp = Exp_Main_Multi
    is_training = True
    if args.is_training:
        for ii in range(args.itr): ## iteration
            paramgrids = {
                'Input': {
                    'seq_len_L': {'type': int, 'values': [48, 60, 72, 84, 96, 108, 120, 132]},
                    'seq_len_S': {'type': int, 'values': [3, 6, 9, 12, 15, 18, 21, 24]},
                    'label_len': {'type': int, 'values': [8, 16, 24, 32, 40, 48, 56, 72]},
                },
                'Model': {
                    'e_layers': {'type': int, 'values': [1, 2, 3, 4, 5]},
                    'd_layers': {'type': int, 'values': [1, 2, 3, 4, 5]},
                    'd_model': {'type': int, 'values': [256, 512, 1024]},
                    'n_heads': {'type': int, 'values': [2, 4, 8, 16]},
                    'moving_avg_L': {'type': int, 'values': [4, 8, 12, 16]},
                    'moving_avg_S': {'type': int, 'values': [3, 6, 9]}
                },
                'Training': {
                    'batch_size': {'type': int, 'values': [32, 128, 256, 512]},
                    'learning_rate': {'type': float, 'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-5]}
                }
            }

            if args.search_method in ["GridSearch", "RandomSearch"]:
                search_algorithm = Graidsearch if args.search_method == "GridSearch" else Randomsearch

                best_params_all = dict()
                for group_name, paramgrid in paramgrids.items():
                    print(f"Running {args.search_method} for {group_name} group")
                    counts = [len(value['values']) for value in paramgrid.values()]
                    Max_iteration = reduce(lambda x, y: x * y, counts)  # 모든 조합의 수
                    best_params, best_score = search_algorithm(paramgrid, Exp, args,Max_iteration)
                    best_params_all[group_name] = best_params  # 결과 저장

                best_params_all['best_score'] = float(best_score)  # 최종 손실 값 저장

            elif args.search_method in ["BayesianSearch"]: # BayesianSearch
                search_algorithm = BayesianOptimization

                best_params_all = dict()
                for group_name, paramgrid in paramgrids.items():
                    print(f"Running {args.search_method} for {group_name} group")
                    counts = [len(value['values']) for value in paramgrid.values()]
                    Max_iteration = reduce(lambda x, y: x * y, counts)  # 모든 조합의 수
                    best_params, best_score = search_algorithm(paramgrid, Exp, args, Max_iteration)
                    best_params_all[group_name] = best_params  # 결과 저장

                best_params_all['best_score'] = float(best_score)  # 최종 손실 값 저장

            # save_best_params
            with open(f"{args.search_method}.json", "w", encoding="utf-8") as json_file:
                json.dump(best_params_all, json_file, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                print("All best parameters:", best_params_all)

        if 'best_score' in best_params_all:
            del best_params_all['best_score']

        for keys in best_params_all.keys():
            for key, value in best_params_all[keys].items():
                setattr(args, key, value)

    setting = (f'{args.model_id}_'
               f'{args.model}_'
               f'{args.data}_'
               f'ft{args.seq_len_L}_'
               f'sl{args.seq_len_S}_'
               f'll{args.label_len}_'
               f'pl{args.label_len}_'
               f'dm{args.e_layers}_'
               f'nh{args.d_layers}_'
               f'el{args.d_model}_'
               f'dl{args.moving_avg_L}_'
               f'df{args.moving_avg_S}_'
               f'fc{args.batch_size}_'
               f'eb{args.learning_rate}_')

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp = Exp(args)
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    #seed = 2021, 42, 72021, 20215, 921
    import warnings
    warnings.filterwarnings('ignore')

    root = './Data'

    a = pd.read_csv(f'./Data/station_1.csv')
    main(seed=2021, num='Total', target=f'{a.columns[2]}', model='DeFusionformer', root=root, types='test', search_method="RandomSearch")
    main(seed=2021, num='Total', target=f'{a.columns[2]}', model='DeFusionformer', root=root, types='test', search_method="BayesianSearch")
    main(seed=2021, num='Total', target=f'{a.columns[2]}', model='DeFusionformer', root=root, types='test', search_method="GridSearch")