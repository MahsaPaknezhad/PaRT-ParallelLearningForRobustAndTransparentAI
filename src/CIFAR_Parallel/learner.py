import os
import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Learner():
    def __init__(self, model, args, sessions, trainloaders, testloaders, labels, class_per_tasks, logit_init_inds,
                 use_cuda, train_paths):
        self.model = model
        self.args = args
        self.title = 'CIFAR_' + self.args['arch']
        self.sessions = sessions
        self.trainloaders = trainloaders
        self.use_cuda = use_cuda
        self.testloaders = testloaders
        self.start_epoch = self.args['start_epoch']
        self.test_loss = 0.0
        self.train_paths = train_paths
        self.test_acc = 0.0
        self.train_loss, self.train_acc = 0.0, 0.0
        self.alllabels = labels
        self.class_per_tasks = class_per_tasks
        self.logits_init_inds = logit_init_inds
        self.num_batches = []
        self.mapped_labels = []
        for i in range(self.args['num_tasks']):
            self.mapped_labels.append(range(self.class_per_tasks[i]))
            self.num_batches.append(len(self.trainloaders[i]))

        print(self.num_batches)
        self.trainable_params = []
        self.optimizers = []
        params_set = [self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4, self.model.conv5,
                      self.model.conv6, self.model.conv7, self.model.conv8, self.model.conv9]
        for k in range(len(self.sessions)):
            t_params = []
            for j, params in enumerate(params_set):
                for i, param in enumerate(params):
                    if (self.train_paths[k][j, i] == 1):
                        p = {'params': param.parameters()}
                        t_params.append(p)
                    else:
                        param.requires_grad = False

            p = {'params': self.model.final_layer.parameters()}
            t_params.append(p)
            print("Number of layers being trained for task %d : %d"% (k, len(t_params)))
            self.trainable_params.append(t_params)
            optimizer = optim.Adam(t_params, lr=self.args['lr'], betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=0,
                                   amsgrad=False)
            self.optimizers.append(optimizer)

    def learn(self):
        logger = Logger(os.path.join(self.args['checkpoint'],
                                     '_' + str(self.args['test_case']) + '_log.txt'), title=self.title)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

        for epoch in range(self.args['start_epoch'], self.args['epochs']):
            self.adjust_learning_rate(epoch)
            self.train_iters = [iter(trainloader) for trainloader in self.trainloaders]
            self.train(epoch)
            self.test(epoch)

            # append logger file
            logger.append([epoch, self.args['lr'], self.train_loss, self.test_loss, self.train_acc, self.test_acc])

            # save model
            if (epoch + 1) % self.args['log_interval'] == 0:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'acc': self.test_acc,
                    'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
                }, checkpoint=self.args['savepoint'], epoch=epoch, test_case=self.args['test_case'])

        logger.close()
        logger.plot()

        savefig(os.path.join(self.args['checkpoint'], 'log_' + str(self.args['test_case']) + '.eps'))

    def train(self, epoch):
        # switch to train mode
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        more_data = True
        num_read_batches = [0] * len(self.sessions)
        n_batches = 10
        bar = Bar('Processing')
        while more_data:
            more_data = False
            np.random.shuffle(self.sessions)
            for k in self.sessions:
                n = min(n_batches, self.num_batches[k] - num_read_batches[k])
                if n > 0:
                    more_data = True
                    num_read_batches[k] += n

                for i in range(n):
                    (inputs, targets) = next(self.train_iters[k])
                    targets = self.map_labels(targets, k)
                    if self.use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                    # compute output
                    outputs = self.model(inputs, self.train_paths[k])
                    tar_ce = targets
                    pre_ce = outputs.clone()

                    pre_ce = pre_ce[:, self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]]

                    loss = F.cross_entropy(pre_ce, tar_ce)

                    # measure accuracy and record loss
                    if (self.args['dataset'][k] == 'cifar10'):
                        prec1, prec5 = accuracy(output=outputs.data[:,
                                                       self.logits_init_inds[k]:self.logits_init_inds[k] +
                                                                                self.class_per_tasks[k]],
                                                target=targets.data, topk=(1, 2))
                    else:
                        prec1, prec5 = accuracy(output=outputs.data[:,
                                                       self.logits_init_inds[k]:self.logits_init_inds[k] +
                                                                                self.class_per_tasks[k]],
                                                target=targets.data, topk=(1, 5))
                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                    # compute gradient and do SGD step
                    self.optimizers[k].zero_grad()
                    loss.backward()
                    self.optimizers[k].step()

        # plot progress
        bar.suffix = '( Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
            total=bar.elapsed_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg
        )
        bar.next()
        bar.finish()
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, self.args['epochs'], self.args['lr']))
        self.train_loss, self.train_acc = losses.avg, top1.avg

    def test(self, epoch):

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        bar = Bar('Processing')
        self.sessions = list(range(self.args['num_tasks']))
        for k in self.sessions:
            for batch_idx, (inputs, targets) in enumerate(self.testloaders[k]):
                # measure data loading time

                targets = self.map_labels(targets, k)
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                outputs = self.model(inputs, self.train_paths[k])
                tar_ce = targets
                pre_ce = outputs.clone()

                pre_ce = pre_ce[:, self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]]
                loss = F.cross_entropy(pre_ce, tar_ce)

                # measure accuracy and record loss
                if (self.args['dataset'][k] == 'cifar10'):
                    prec1, prec5 = accuracy(
                        outputs.data[:, self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]],
                        targets.data, topk=(1, 2))
                else:
                    prec1, prec5 = accuracy(
                        outputs.data[:, self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]],
                        targets.data, topk=(1, 5))

                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

            # plot progress
            bar.suffix = '(Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg
            )
            bar.next()
        bar.finish()
        print('epoch: %d val loss: %0.03f val acc: %0.03f losses size: %d top1 size: %d' % (
        epoch, losses.avg, top1.avg, losses.count, top1.count))
        self.test_loss = losses.avg
        self.test_acc = top1.avg

    def save_checkpoint(self, state, checkpoint='checkpoint', filename='checkpoint.pth.tar', epoch=0, test_case=0):
        torch.save(state, os.path.join(checkpoint, 'model_' + str(test_case) + '_epoch_' + str(epoch) + '.pth.tar'))

    def adjust_learning_rate(self, epoch):
        if epoch in self.args['schedule']:
            self.args['lr'] *= self.args['gamma']
            for i in range(len(self.optimizers)):
                for param_group in self.optimizers[i].param_groups:
                    param_group['lr'] = self.args['lr']

    def get_confusion_matrix(self, k):

        confusion_matrix = torch.zeros(self.class_per_tasks[k], self.class_per_tasks[k])
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloaders[k]):
                targets = self.map_labels(targets, k)
                if self.args['use_cuda']:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.model(inputs, self.train_paths[k])
                pre_ce = outputs.clone()
                pre_ce = pre_ce[:, self.logits_init_inds[k]:self.logits_init_inds[k] + self.class_per_tasks[k]]
                _, preds = torch.max(pre_ce, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        return confusion_matrix

    def map_labels(self, targets, k):
        for n, i in enumerate(self.alllabels[k]):
            targets[targets == i] = self.mapped_labels[k][n]
        return targets
