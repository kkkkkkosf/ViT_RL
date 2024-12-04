import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from variables import *
import numpy as np
import os
import time
import argparse
import random
from dataloader import SOXSeT_CSV,EVBSeT_CSV
import torch.utils.data as Data
import pandas as pd
from utils import Bar, AverageMeter, accuracy
import torch.nn.functional as F
from sklearn import metrics
import torch
from rl_models import Generator, CriticModel
from ppo import GAE, PPO_step
from vit_model import ViT
from torchsummary import summary
import statsmodels.api as sm

parser = argparse.ArgumentParser(description="Hyper-Parameters")
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--lr", "--learning-rate", default=0.01, type=float)
parser.add_argument("--lr_models", "--learning-rate-rl", default=4e-5, type=float)
parser.add_argument('--schedule', type=int, default=[100, 200, 300,400])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float)
parser.add_argument('--manualSeed', type=int, help='manual seed')

hy_args = parser.parse_args()
state = {k: v for k, v in hy_args._get_kwargs()}

os.environ['CUDA_VISIBLE_DEVICES'] = gpus
use_cuda = torch.cuda.is_available()
if hy_args.manualSeed is None:
    hy_args.manualSeed = random.randint(1, 10000)
random.seed(hy_args.manualSeed)
torch.manual_seed(hy_args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(hy_args.manualSeed)

best_acc = 0.0
best_auc = 0.0
eval_key = True
save_name_pre = "{}_{}".format(save_number, state["lr"])


class Training_procedure():

    def __init__(self, args, model, generator, critic):
        SOX_loader_train = EVBSeT_CSV("path", "train")
        SOX_loader_test = EVBSeT_CSV("path","test")


        self.trainloader = Data.DataLoader(dataset=SOX_loader_train, batch_size=batchsize_train, shuffle=True,
                                            num_workers=4, drop_last=False)
        self.testloader = Data.DataLoader(dataset=SOX_loader_test, batch_size=batchsize_test, shuffle=False,
                                            num_workers=4, drop_last=False)
        self.model = model
        self.generator = generator
        self.critic = critic
        self.state = {k: v for k, v in args._get_kwargs()}
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=args.lr_models)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=args.lr_models)

    def training_iterations(self, epochs):
        results_vit = {'test_loss': [], 'train_loss': [], 'acc': [], 'sen': [], 'spe': [], 'auc': [], 'best acc': [],
                   'best auc': [],'lr': [],'train_top1':[],'test_top1':[]}
        results_rl = {'test_loss': [],'acc': [], 'sen': [], 'spe': [], 'auc': [], 'best acc': [],
                       'best auc': [], 'lr': [],'top1':[]}
        global best_acc, best_auc,state

        for epoch in range(epochs):
            soft_value = random.uniform(1e-1, 1)
            self.adjust_learning_rate(self.optimizer, epoch)
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs, state['lr']))
            #vit
            loss_train, top1 = self.train_model(self.trainloader, self.model, self.generator, self.criterion, self.optimizer, soft_value)
            loss_test, acc, auc, statistic_sensitivity, statistic_specificity,vit_top1 = self.test(self.testloader, self.model, self.generator, self.criterion, soft_value)
            is_best_acc = acc > best_acc
            is_best_auc = auc > best_auc
            if is_best_acc:
                best_acc = max(acc, best_acc)
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer,
                    "generator": self.generator.state_dict(),
                    "critic": self.critic.state_dict(),
                    "optimizer_static": self.optimizer_critic.state_dict(),
                    "optimizer_generator": self.optimizer_gen.state_dict(),
                    "soft_value": soft_value,
                    "epoch": epoch
                }
                torch.save(state_dict, "./pths_new/best_acc_vit_{}.pth".format(save_number))
            if is_best_auc:
                best_auc = max(auc, best_auc)
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer,
                    "generator": self.generator.state_dict(),
                    "critic": self.critic.state_dict(),
                    "optimizer_static": self.optimizer_critic.state_dict(),
                    "optimizer_generator": self.optimizer_gen.state_dict(),
                    "soft_value": soft_value,
                    "epoch": epoch
                }
                torch.save(state_dict, "./pths_new/best_auc_vit_{}.pth".format(save_number))

            print(
                "Vit acc: {}, auc: {}, sen: {}, spe: {}, best acc: {}, best auc: {}".format(acc, auc, statistic_sensitivity,
                                                                                        statistic_specificity, best_acc,
                                                                                        best_auc))

            results_vit['acc'].append(acc)
            results_vit['sen'].append(statistic_sensitivity)
            results_vit['spe'].append(statistic_specificity)
            results_vit['auc'].append(auc)
            results_vit['lr'].append(state['lr'])
            results_vit['best auc'].append(best_auc)
            results_vit['best acc'].append(best_acc)
            results_vit['train_loss'].append(loss_train)
            results_vit['test_loss'].append(loss_test)
            results_vit['train_top1'].append(top1)
            results_vit['test_top1'].append(vit_top1)

            data_frame = pd.DataFrame(data=results_vit)
            data_frame.to_csv('result_new/{}_vit.csv'.format(save_name_pre), index_label='epoch')


            # rl
            self.train_rl(self.trainloader, self.model, self.generator, self.critic, self.optimizer_gen, self.optimizer_critic, soft_value)

            loss_test, acc, auc, statistic_sensitivity, statistic_specificity,rl_top1 = self.test(self.testloader, self.model, self.generator, self.criterion, soft_value)
            is_best_acc = acc > best_acc
            is_best_auc = auc > best_auc
            if is_best_acc:
                best_acc = max(acc, best_acc)
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer,
                    "generator": self.generator.state_dict(),
                    "critic": self.critic.state_dict(),
                    "optimizer_static": self.optimizer_critic.state_dict(),
                    "optimizer_generator": self.optimizer_gen.state_dict(),
                    "soft_value": soft_value,
                    "epoch": epoch
                }
                torch.save(state_dict, "./pths_new/best_acc_rl_{}.pth".format(save_number))
            if is_best_auc:
                best_auc = max(auc, best_auc)
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer,
                    "generator": self.generator.state_dict(),
                    "critic": self.critic.state_dict(),
                    "optimizer_static": self.optimizer_critic.state_dict(),
                    "optimizer_generator": self.optimizer_gen.state_dict(),
                    "soft_value": soft_value,
                    "epoch": epoch
                }
                torch.save(state_dict, "./pths_new/best_auc_rl_{}.pth".format(save_number))
            print("RL acc: {}, auc: {}, sen: {}, spe: {}, best acc: {}, best auc: {}".format(acc, auc, statistic_sensitivity, statistic_specificity, best_acc, best_auc))

            results_rl['acc'].append(acc)
            results_rl['sen'].append(statistic_sensitivity)
            results_rl['spe'].append(statistic_specificity)
            results_rl['auc'].append(auc)
            results_rl['lr'].append(state['lr'])
            results_rl['best auc'].append(best_auc)
            results_rl['best acc'].append(best_acc)
            results_rl['test_loss'].append(loss_test)
            results_rl['top1'].append(rl_top1)

            data_frame = pd.DataFrame(data=results_rl)
            data_frame.to_csv('result_new/{}_rl.csv'.format(save_name_pre), index_label='epoch')


    def train_model(self, trainloader, model, generator, criterion, optimizer, soft_value):
        print("train_model")
        losses = AverageMeter()
        top1 = AverageMeter()
        model.train()
        for key, param in model.named_parameters():
            param.requires_grad = True

        for _, p in generator.named_parameters():
            p.requires_grad = False
        bar = Bar('Processing', max=len(trainloader))
        for batch_idx, batch_data in enumerate(trainloader):
            inputs = batch_data["img"].float()
            targets = batch_data["label"].float()
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs, _, _ = model(inputs, soft_value, True)
            probabilities = torch.softmax(outputs,dim=-1)
            probability = torch.max(probabilities)
            loss_penalty = 0.1 * torch.abs(probability - 0.5)
            loss_1 = criterion(outputs, targets.long())
            loss_2 = - loss_penalty
            loss = loss_1 + loss_2
            prec1 = accuracy(outputs.data, targets.long().data, topk=(1,))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0].item(), inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.suffix = 'Vit ({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                batch=batch_idx + 1,
                size=len(trainloader),
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
            )
            bar.next()
        bar.finish()
        return (losses.avg, top1.avg)

    def train_rl(self, trainloader, model, generator, critic, optimizer_gen, optimizer_critic, soft_value):
        print("train_rl")
        v_loss_ = AverageMeter()
        p_loss_ = AverageMeter()
        reward_ = AverageMeter()
        model.eval()

        for key, param in model.named_parameters():
            param.requires_grad = False
        for _, p in generator.named_parameters():
            p.requires_grad = True

        bar = Bar('Processing', max=len(trainloader))
        for batch_idx, batch_data in enumerate(trainloader):
            inputs = batch_data["img"].float()
            targets = batch_data["label"].float()
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            with torch.no_grad():
                outputs, action, state = model(inputs, soft_value, True)
                value_o = critic(action)
                fixed_log_prob = generator.module.get_log_prob(state, soft_value)
            prec1 = accuracy(outputs.data, targets.long().data, topk=(1,))
            prec1_value = prec1[0] / 100
            gen_r = prec1_value - 0.8
            gen_r = torch.tensor([gen_r] * inputs.size(0))

            advantages, returns = GAE(-torch.log(2 - gen_r), value_o, gamma=0.1, lam=0.1)

            v_loss, p_loss = PPO_step(generator, critic, optimizer_gen, optimizer_critic, state,
                                      action, returns, advantages, fixed_log_prob, 1e-5, 1e-5, soft_value)
            v_loss_.update(v_loss.item(), 1)
            p_loss_.update(p_loss.item(), 1)
            reward_.update(returns.cpu().mean().numpy() * (-1), 1)
            bar.suffix = 'RL ({batch}/{size}) | Total: {total:} | ETA: {eta:} | v_loss: {v_loss:.4f} | p_loss: {p_loss:.4f} | reward: {reward:.3f}'.format(
                batch=batch_idx + 1,
                size=len(trainloader),
                total=bar.elapsed_td,
                eta=bar.eta_td,
                v_loss=v_loss_.avg,
                p_loss=p_loss_.avg,
                reward=reward_.avg
            )
            bar.next()
        bar.finish()

    def test(self, testloader, model, generator, criterion, soft_value):
        losses = AverageMeter()
        top1 = AverageMeter()
        model.eval()
        for key, param in model.named_parameters():
            param.requires_grad = False
        for _, p in generator.named_parameters():
            p.requires_grad = False
        bar = Bar('Processing', max=len(testloader))
        people_id = []
        pred = []
        labels = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(testloader):
                inputs = batch_data["img"].float()
                targets = batch_data["label"].float()
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs, action, state = model(inputs, soft_value, False)
                people_id.extend(batch_data['id'])
                pred.extend(F.softmax(outputs, -1)[:, 1].detach().cpu().numpy().tolist())
                labels.extend(targets.detach().cpu().numpy().tolist())
                loss = criterion(outputs, targets.long())
                prec1 = accuracy(outputs.data, targets.long().data, topk=(1,))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0].item(), inputs.size(0))
                bar.suffix = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                )

                bar.next()
            bar.finish()
            df = pd.DataFrame({'people_id': people_id, 'preds': pred, 'labels': labels})
            acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
            single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
            statistic_sensitivity, statistic_specificity,total_preds,total_label,acc_ic,sen_ic,spe_ic = self.auc(df)
        return (losses.avg, acc_statistic, statis, statistic_sensitivity, statistic_specificity,top1.avg)



    def auc(self, df):
        def threshold(ytrue, ypred):
            fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred)
            y = tpr - fpr
            youden_index = np.argmax(y)
            optimal_threshold = thresholds[youden_index]
            point = [fpr[youden_index], tpr[youden_index]]
            roc_auc = metrics.auc(fpr, tpr)
            return optimal_threshold, point, fpr, tpr, roc_auc
        single_threshold, single_point, single_fpr, single_tpr, single = threshold(df['labels'], df['preds'])
        df['single'] = (df['preds'] >= single_threshold).astype(int)
        acc_single = (df['labels'] == df['single']).mean()
        df = df.groupby('people_id')[['labels', 'preds']].mean()

        statistic_threshold, statistic_point, statistic_fpr, statistic_tpr, statis = threshold(df['labels'], df['preds'])
        df['outputs'] = (df['preds'] >= statistic_threshold).astype(int)
        acc_statistic = (df['labels'] == df['outputs']).mean()
        df_sensitivity = df.loc[df["labels"] == 1]
        statistic_sensitivity = (df_sensitivity['labels'] == df_sensitivity['outputs']).mean()
        df_specificity = df.loc[df["labels"] == 0]
        statistic_specificity = (df_specificity['labels'] == df_specificity['outputs']).mean()
        total_num = len(df['labels'])
        right_num = (df['labels'] == df['outputs']).sum()
        acc_ic = sm.stats.proportion_confint(right_num, total_num, alpha=0.05, method='normal')
        sen_num = (df_sensitivity['labels'] == df_sensitivity['outputs']).sum()
        sen_ic = sm.stats.proportion_confint(sen_num, total_num, alpha=0.05, method='normal')
        spe_num = (df_specificity['labels'] == df_specificity['outputs']).sum()
        spe_ic = sm.stats.proportion_confint(spe_num, total_num, alpha=0.05, method='normal')
        return acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
               single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
               statistic_sensitivity, statistic_specificity,df['preds'],df['labels'],acc_ic,sen_ic,spe_ic

    def adjust_learning_rate(self, optimizer, epoch):
        global state
        if epoch in hy_args.schedule:
            state['lr'] *= hy_args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']

if __name__ == '__main__':
    critic = CriticModel(dim_state_action=768, dim_hidden=512, dim_out=1).cuda()
    generator = Generator(dim_state=768, dim_hidden1=1024, dim_hidden2=1024, dim_action=768).cuda()
    model = ViT('B_16', pretrained=False, num_classes=2, image_size=image_size,generator = generator)
    model = torch.nn.DataParallel(model).cuda()
    generator = torch.nn.DataParallel(generator).cuda()
    critic = torch.nn.DataParallel(critic).cuda()
    soft_value = 0.01
    total_params = sum(p.numel() for p in model.parameters())
    print('total_params: {}'.format(total_params))
    cudnn.benchmark = True
    Training_steps = Training_procedure(hy_args, model, generator, critic)
    Training_steps.training_iterations(hy_args.epochs)
