import logging
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DeFusionformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main_Multi(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_Multi, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DeFusionformer': DeFusionformer,
        }
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        def _run_model():
            outputs = self.model(batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion, setting):
        total_loss = []
        preds = [];trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark) in enumerate(vali_loader):
                batch_x_L = batch_x_L.float().to(self.device)
                batch_x_S = batch_x_S.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_L_mark = batch_x_L_mark.float().to(self.device)
                batch_x_S_mark = batch_x_S_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark)


                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

                ################## 하이퍼#################
                preds.append(pred)
                trues.append(true)
                ################## 하이퍼#################

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        SC = test_loader.dataset.scaler

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss_V1 = []
        vali_loss_V1 = []

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x_L = batch_x_L.float().to(self.device)
                batch_x_S = batch_x_S.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_L_mark = batch_x_L_mark.float().to(self.device)
                batch_x_S_mark = batch_x_S_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print(f"\titers: {i + 1:.4f}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            #원래는 해당라인
            # print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, setting)
            test_loss = self.vali(test_data, test_loader, criterion, setting)
            #### 원래는 해당라인
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} test Loss: {test_loss:.7f}")
            train_loss_V1.append(train_loss)
            vali_loss_V1.append(vali_loss)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        SC = test_loader.dataset.scaler
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark) in enumerate(test_loader):
                batch_x_L = batch_x_L.float().to(self.device)
                batch_x_S = batch_x_S.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_L_mark = batch_x_L_mark.float().to(self.device)
                batch_x_S_mark = batch_x_S_mark.float().to(self.device)


                outputs, batch_y = self._predict(batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                true = SC.inverse_transform(np.squeeze(true).reshape(-1, 1))
                pred = SC.inverse_transform(np.squeeze(pred).reshape(-1, 1))

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)


        # result save
        folder_path = './Results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, nrmse = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, rmse:{rmse}, nrmse:{nrmse}')
        f = open("results.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'mse:{mse}, mae:{mae}, rmse:{rmse}')
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, nrmse]))
        np.save(folder_path + 'pred_real.npy', preds)
        np.save(folder_path + 'true_real.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark) in enumerate(pred_loader):
                batch_x_L = batch_x_L.float().to(self.device)
                batch_x_S = batch_x_S.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_L_mark = batch_x_L_mark.float().to(self.device)
                batch_x_S_mark = batch_x_S_mark.float().to(self.device)


                outputs, batch_y = self._predict(batch_x_L, batch_x_L_mark, batch_x_S, batch_x_S_mark, batch_y, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        return
