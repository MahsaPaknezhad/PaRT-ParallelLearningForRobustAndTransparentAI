'''
Copyright (c) Mahsa Paknezhad, 2021
'''

import os
import torch
from utils import Bar, Logger, AverageMeter, accuracy
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Learner():
    def __init__(self,model,args, sessions, trainloaders, testloaders, class_per_tasks, logit_init_inds, use_cuda, train_paths):
        self.model=model
        self.args=args
        self.title= '5tasks_' + self.args["arch"]
        self.sessions = sessions
        self.trainloaders = trainloaders
        self.use_cuda=use_cuda
        self.testloaders = testloaders
        self.start_epoch=self.args["start_epoch"]
        self.train_paths = train_paths
        self.class_per_tasks = class_per_tasks
        self.logits_init_inds = logit_init_inds
        self.num_batches = []
        self.trainable_params = []
        self.optimizers = []
        for i in range(self.args["num_tasks"]):
            self.num_batches.append(len(self.trainloaders[i]))
        print('num of batches per task : ', self.num_batches)
        self.n_batches = min(50, min(self.num_batches))

        # set modules specified in path to be trainable and freeze the rest of the modules
        if (self.args["arch"] in  ["resnet50", "resnet18"] ):
            params_set = [self.model.convs1,  self.model.bns1, self.model.layers1, self.model.layers2, self.model.layers3, self.model.layers4]
        for k in range(len(self.sessions)):
            t_params = []
            for j, params in enumerate(params_set):
                if j>0:
                    j -= 1
                for i, param in enumerate(params):
                    if isinstance(param, list):
                        param = param[k]
                    if (self.train_paths[k][j, i] == 1):
                        p = {'params': param.parameters()}
                        t_params.append(p)
                    else:
                        param.requires_grad = False

            p = {'params': self.model.fc.parameters()}
            t_params.append(p)
            print("Number of layers being trained : ", len(t_params))
            self.trainable_params.append(t_params)
            optimizer = optim.Adam(t_params, lr=self.args["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            self.optimizers.append(optimizer)
        self.loggers = []

        for i in self.sessions:
            self.loggers.append(Logger(os.path.join(self.args["checkpoint"], '_' + str(self.args["test_case"]) + '_' + str(i) + '_log.txt'),
                             title=self.title))
            self.loggers[-1].set_names(['Epoch',  'Learning Rate', 'Train Loss', 'Train Acc', 'Valid Loss', 'Valid Acc'])



    def learn(self):
        # trains and tests the base network on the specified tasks

        for epoch in range(self.args["start_epoch"], self.args["num_epochs"]):
            self.adjust_learning_rate(epoch)
            self.train_iters = [iter(trainloader) for trainloader in self.trainloaders]

            self.train()
            self.measure_train_accuracy()
            self.test(epoch)

            self.plot(os.path.join(self.args["checkpoint"], 'log_' + str(self.args["test_case"]) + '.eps'))

            # save model
            if (epoch + 1) % self.args["log_interval"] == 0:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
                }, checkpoint=self.args["savepoint"], epoch=epoch, test_case=self.args["test_case"])

        for i in self.sessions:
            self.loggers[i].close()



    def train(self):
        # trains the base network on each task in parallel. Each task will be trained on its assigned sequence of modules (define as path)
        self.model.train()
        more_data = True
        num_read_batches = [0] * len(self.sessions)

        while more_data:
            more_data = False
            np.random.shuffle(self.sessions)
            n = max([min(self.n_batches, self.num_batches[k] - num_read_batches[k]) for k in self.sessions])
            if n > 0:
                more_data = True
                for k in self.sessions:
                    num_read_batches[k] += n
                    for i in range(n):
                        try:
                            (inputs, targets) = next(self.train_iters[k])
                        except StopIteration:
                            self.train_iters[k] = iter(self.trainloaders[k])
                            (inputs, targets) = next(self.train_iters[k])

                        if self.use_cuda:
                            inputs, targets = inputs.cuda(), targets.cuda()
                        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                        # compute output
                        outputs = self.model(inputs, self.train_paths[k], k)
                        tar_ce = targets
                        pre_ce = outputs.clone()

                        pre_ce = pre_ce[:, self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]]

                        loss = F.cross_entropy(pre_ce, tar_ce)
                        # compute gradient and do SGD step
                        self.optimizers[k].zero_grad()
                        loss.backward()
                        self.optimizers[k].step()
                        more_data=False
                        break

    def measure_train_accuracy(self):
        # measures the train accuracy of the base network on each task

        self.losses = [AverageMeter() for i in self.sessions]
        self.top1 = [AverageMeter() for i in self.sessions]
        self.top5 = [AverageMeter() for i in self.sessions]

        self.model.eval()
        #print('Test session:', self.sessions[k])
        self.sessions = list(range(self.args["num_tasks"]))
        for k in self.sessions:
            for batch_idx, (inputs, targets) in enumerate(self.trainloaders[k]):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(),targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = self.model(inputs, self.train_paths[k], k)
                tar_ce = targets
                pre_ce = outputs.clone()

                pre_ce = pre_ce[:,  self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]]
                loss = F.cross_entropy(pre_ce, tar_ce)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data[:, self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]], targets.data, topk=(1, 5))

                self.losses[k].update(loss.item(), inputs.size(0))
                self.top1[k].update(prec1.item(), inputs.size(0))
                self.top5[k].update(prec5.item(), inputs.size(0))
    
    def test(self, epoch):
        # measures the validation accuracy of the base network on each task using the assigned sequence of modules (path) to the task
        self.model.eval()

        self.sessions = list(range(self.args["num_tasks"]))
        for k in self.sessions:
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            for batch_idx, (inputs, targets) in enumerate(self.testloaders[k]):
                # measure data loading time

                if self.use_cuda:
                    inputs, targets = inputs.cuda(),targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = self.model(inputs, self.train_paths[k], k)
                tar_ce = targets
                pre_ce = outputs.clone()

                pre_ce = pre_ce[:,  self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]]
                loss = F.cross_entropy(pre_ce, tar_ce)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data[:, self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]], targets.data, topk=(1, 5))

                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

            self.loggers[k].append([epoch, self.args['lr'], self.losses[k].avg,self.top1[k].avg,  losses.avg, top1.avg])
            print('epoch: %d task: %d val loss: %0.03f  val acc: %0.03f losses size: %d top1 size: %d' % (
                epoch, k, losses.avg, top1.avg, losses.count, top1.count))


    def save_checkpoint(self,state, checkpoint='checkpoint', filename='checkpoint.pth.tar',epoch=0, test_case=0):
        # save the current base network
        torch.save(state, os.path.join(checkpoint, 'model_'+str(test_case)+'_epoch_' + str(epoch)+'.pth.tar'))


    def adjust_learning_rate(self, epoch):
        # update the learning rate of the optimizer
        if epoch in self.args["schedule"]:
            self.args['lr'] *= self.args["gamma"]
            for i in range(len(self.optimizers)):
                for param_group in self.optimizers[i].param_groups:
                    param_group['lr'] = self.args['lr']



    def get_confusion_matrix(self, k):
        # build the confusion matrix for the task k
        confusion_matrix = torch.zeros(self.class_per_tasks[k], self.class_per_tasks[k])
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloaders[k]):
                if self.args["use_cuda"]:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs, self.train_paths[k], k)
                pre_ce = outputs.clone()
                pre_ce = pre_ce[:, self.logits_init_inds[k] :self.logits_init_inds[k]+self.class_per_tasks[k]]
                _, preds = torch.max(pre_ce, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        return confusion_matrix

    def plot(self, fname):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
        for k in self.sessions:
            numbers = self.loggers[k].numbers
            names = self.loggers[k].names
            for i, name in enumerate([names[3]]):
                x = np.arange(len(numbers[name]))
                ax1.plot(x, np.asarray(numbers[name]), label=name+'_%d'%k)
        ax1.legend(loc='upper left')
        ax1.grid(True)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        for k in self.sessions:
            numbers = self.loggers[k].numbers
            names = self.loggers[k].names
            for i, name in enumerate([names[2]]):
                x = np.arange(len(numbers[name]))
                ax3.plot(x, np.asarray(numbers[name]), label=name+'_%d'%k)
        ax3.legend(loc='upper left')
        ax3.grid(True)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        for k in self.sessions:
            numbers = self.loggers[k].numbers
            names = self.loggers[k].names
            for i, name in enumerate([names[5]]):
                x = np.arange(len(numbers[name]))
                ax2.plot(x, np.asarray(numbers[name]), label=name+'_%d'%k)
        ax2.legend(loc='upper left')
        ax2.grid(True)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        for k in self.sessions:
            numbers = self.loggers[k].numbers
            names = self.loggers[k].names
            for i, name in enumerate([names[4]]):
                x = np.arange(len(numbers[name]))
                ax4.plot(x, np.asarray(numbers[name]), label=name+'_%d'%k)
        ax4.legend(loc='upper left')
        ax4.grid(True)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        plt.savefig(fname, dpi=150)



