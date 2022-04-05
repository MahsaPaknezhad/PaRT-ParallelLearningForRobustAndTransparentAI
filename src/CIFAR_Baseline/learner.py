'''
Copyright (c) Mahsa Paknezhad, 2021
'''

import os
from tqdm import tqdm
import torch
from utils import Bar, Logger, AverageMeter, accuracy, savefig
import torch.optim as optim
import time
import torch.nn.functional as F


class Learner():
    def __init__(self,model,args,trainloader, testloader,labels, class_per_task, logit_init_ind, use_cuda, path, train_path, infer_path):
        self.model=model
        self.args=args
        self.title= 'CIFAR_' + self.args['arch']
        self.trainloader=trainloader 
        self.use_cuda=use_cuda
        self.testloader=testloader
        self.start_epoch=self.args['start_epoch']
        self.test_loss=0.0
        self.path = path
        self.train_path = train_path
        self.infer_path = infer_path
        self.test_acc=0.0
        self.train_loss, self.train_acc=0.0,0.0
        self.labels = labels
        self.mapped_labels = range(class_per_task)
        self.class_per_task = class_per_task
        self.logit_init_ind = logit_init_ind

        trainable_params = []

        # set modules specified in path to be trainable and freeze the rest of the modules
        params_set = [self.model.conv1, self.model.conv2, self.model.conv3, self.model.conv4, self.model.conv5, self.model.conv6, self.model.conv7, self.model.conv8, self.model.conv9]
        for j, params in enumerate(params_set): 
            for i, param in enumerate(params):
                if(self.train_path[j,i]==1):
                    p = {'params': param.parameters()}
                    trainable_params.append(p)
                else:
                    param.requires_grad = False

        p = {'params': self.model.final_layer.parameters()}
        trainable_params.append(p)
        print("Number of layers being trained : " , len(trainable_params))

        self.optimizer = optim.Adam(trainable_params, lr=self.args['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    def learn(self):
        # trains and tests the base network on the input task
        if self.args['resume']:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(self.args['resume']), 'Error: no checkpoint directory found!'
            self.args['checkpoint'] = os.path.dirname(self.args['resume'])
            checkpoint = torch.load(self.args['resume'])
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(self.args['checkpoint'], 'log_'+str(self.args['task_num'])+'_'+str(self.args['test_case'])+'.txt'), title=self.title, resume=True)
        else:
            logger = Logger(os.path.join(self.args['checkpoint'], 'log_'+str(self.args['task_num'])+'_'+str(self.args['test_case'])+'.txt'), title=self.title)
            logger.set_names(['Task', 'Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        if self.args['evaluate']:
            print('\nEvaluation only')
            self.test(self.start_epoch)
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (self.test_loss, self.test_acc))
            return

        for epoch in range(self.args['start_epoch'], self.args['epochs']):
            self.adjust_learning_rate(epoch)

            print('\nEpoch: [%d | %d] LR: %f Task: %d' % (epoch + 1, self.args['epochs'], self.args['lr'],self.args['task_num']))
            self.train(self.infer_path)
            self.test(self.infer_path)
            # append logger file
            logger.append([self.args['task_num'], epoch, self.args['lr'], self.train_loss, self.test_loss, self.train_acc, self.test_acc])

            # save model
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'acc': self.test_acc,
                    'optimizer' : self.optimizer.state_dict(),
            }, checkpoint=self.args['savepoint'], session=self.args['task_num'], test_case=self.args['test_case'])

        logger.close()
        logger.plot()
        savefig(os.path.join(self.args['checkpoint'], 'log_'+str(self.args['task_num'])+'_'+str(self.args['test_case'])+'.eps'))


    def train(self, path):
        # trains the base network on the input task using the assinged seq of modules (path) to the task
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.trainloader)):
            # measure data loading time
            targets = self.map_labels(targets)

            data_time.update(time.time() - end)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs, path)
            tar_ce = targets
            pre_ce = outputs.clone()

            pre_ce = pre_ce[:, self.logit_init_ind:self.logit_init_ind+self.class_per_task]

            loss = F.cross_entropy(pre_ce, tar_ce)

            # measure accuracy and record loss
            if(self.args['dataset']=='cifar10'):
                prec1, prec5 = accuracy(outputs.data[:,self.logit_init_ind:self.logit_init_ind+self.class_per_task], targets.data, topk=(1, 2))
            else:
                prec1, prec5 = accuracy(outputs.data[:, self.logit_init_ind:self.logit_init_ind + self.class_per_task],targets.data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix  = '({batch}/{size}) | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(self.trainloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.train_loss,self.train_acc=losses.avg, top1.avg

   
    
    def test(self, path):
        # measure test accuracy of the base network on the input task using the assigned sequence of modules (path) to the task
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.testloader)):
            # measure data loading time
            data_time.update(time.time() - end)
            targets = self.map_labels(targets)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            outputs = self.model(inputs, path)

            tar_ce = targets
            pre_ce = outputs.clone()

            pre_ce = pre_ce[:, self.logit_init_ind:self.logit_init_ind+self.class_per_task]

            loss = F.cross_entropy(pre_ce, tar_ce)

            # measure accuracy and record loss
            if(self.args['dataset']=='cifar10'):
                prec1, prec5 = accuracy(outputs.data[:,self.logit_init_ind:self.logit_init_ind+self.class_per_task], targets.data, topk=(1, 2))
            else:
                prec1, prec5 = accuracy(outputs.data[:, self.logit_init_ind:self.logit_init_ind + self.class_per_task],targets.data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.testloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.test_loss= losses.avg;self.test_acc= top1.avg

    def save_checkpoint(self,state, checkpoint='checkpoint', filename='checkpoint.pth.tar',session=0, test_case=0):
        # save the current base network
        torch.save(state, os.path.join(checkpoint, 'task_'+str(session)+'_'+str(test_case)+'_model.pth.tar'))

    def adjust_learning_rate(self, epoch):
        # update the learning rate of the optimizer
        if epoch in self.args['schedule']:
            self.args['lr'] *= self.args['gamma']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args['lr']

    def get_confusion_matrix(self, path):
        # build the confusion matrix for the input task
        confusion_matrix = torch.zeros(self.class_per_task, self.class_per_task)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloader):
                targets = self.map_labels(targets)
                if self.args['use_cuda']:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs, path)
                pre_ce = outputs.clone()
                pre_ce = pre_ce[:,
                         self.logit_init_ind:self.logit_init_ind+self.class_per_task]
                _, preds = torch.max(pre_ce, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        return confusion_matrix

    def map_labels(self, targets):
        # map the randomly selected labels from the CIFAR10/CIFAR100 dataset to a continuous seq of labels
        for n, i in enumerate(self.labels):
            targets[targets==i] = self.mapped_labels[n]
        return targets

