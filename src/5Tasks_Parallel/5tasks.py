'''
Copyright (c) Mahsa Paknezhad, 2021
'''
from __future__ import print_function
import json
import time
import torch.nn.parallel
from utils import Bar,  mkdir_p
from net_utils import get_network
import torch
import random
import argparse
from learner import Learner
from util import *
import sys
import fine_grained_dataset as dataset



def main(args):
    args["checkpoint"] = args["checkpoint"]%(args['dataset'], args["test_case"])
    seed = args["seeds"][int(args["test_case"])]
    print('seed', seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if args["use_cuda"]:
        torch.cuda.manual_seed_all(seed)

    # Create an object of the base network
    model = get_network(args)
    model.load_state_dict(torch.load(os.path.join(args["inputfiles"], 'init_model_new_%s.pth.tar' % str(args["test_case"]))))
    model.train()
    if args["use_cuda"]:
        model = model.cuda()

     # initiate the base network with weights and biases of a pretrained ResNet model trained for ImageNet classification
    if args["use_imagenet_pretrained"]:
        if args["arch"] == 'resnet50' or args["arch"] == 'resnet18':
            state_dict = torch.load(os.path.join(args["modelfile"],args["arch"]+'.pth'))
            cur_state_dict = model.state_dict()
            for name, param in state_dict.items():
                if 'fc' not in name and 'bn' not in name and 'downsample.1' not in name:
                    parts = name.split('.', 1)
                    for j in range(args["M"]):
                        cur_state_dict[parts[0]+str(j)+'.'+parts[1]].copy_(param)
                elif 'bn' in name:
                    parts = name.split('.')
                    if len(parts) == 2:
                        for j in range(args["M"]):
                            for k in range(args["num_tasks"]):
                                cur_state_dict[parts[0]+str(j)+str(k)+'.'+parts[1]].copy_(param)
                    else:
                        for j in range(args["M"]):
                            for k in range(args["num_tasks"]):
                                cur_state_dict[parts[0]+str(j)+'.'+parts[1]+'.'+parts[2]+str(k)+'.'+parts[3]].copy_(param)
                elif 'downsample.1' in name:
                    parts = name.split('.')
                    for j in range(args["M"]):
                        for k in range(args["num_tasks"]):
                            cur_state_dict[parts[0]+str(j)+'.'+parts[1]+'.'+parts[2]+'.'+parts[3]+'.'+str(k)+'.'+parts[4]].copy_(param)
        else:
            print(
                "Currently, we didn't define the mapping of {} between imagenet pretrained weight and our model".format(
                    args["arch"]))
            sys.exit(5)

    print('    Total params: %.2fM' % (sum(p.numel() for n,p in model.named_parameters()) / 1000000.0))

    if not os.path.isdir(args["checkpoint"]):
        mkdir_p(args["checkpoint"])

    args["savepoint"] = args["checkpoint"]

    with open(os.path.join(args["checkpoint"], "setting.json"), 'w') as json_file:
        json.dump(args, json_file, indent = 4, sort_keys=True)

    sessions = list(range(args["num_tasks"]))

    # load the assigned paths (sequence of modules) to the tasks (or ses here)
    train_paths = []
    for ses in sessions:
        path = np.load(args["inputfiles"] + "/path_new_" + str(ses) + "_" + str(args["test_case"]) + ".npy")
        print("train_path %d\n" % ses, path)
        train_paths.append(path)

    print('test case : ' + str(args["test_case"]))
    print('#################################################################################')

    trainloaders = []
    testloaders = []
    logit_init_inds=[0]
    class_per_tasks = []
    logit_init_ind = 0

    # generate the train and validation dataloaders for each task
    for ses in sessions:
        cur_dataset = args["datasets"][ses]
        if 'cropped' in cur_dataset:
            train_loader = dataset.train_loader_cropped(os.path.join(args["datasets_dir"], cur_dataset, 'train'), args["train_batch"], num_workers=args["workers"])
            val_loader = dataset.val_loader_cropped(os.path.join(args["datasets_dir"], cur_dataset, 'test'), args["train_batch"], num_workers=args["workers"])
        else:
            train_loader = dataset.train_loader(os.path.join(args["datasets_dir"], cur_dataset, 'train'), args["train_batch"], num_workers=args["workers"])
            val_loader = dataset.val_loader(os.path.join(args["datasets_dir"], cur_dataset, 'test'), args["train_batch"], num_workers=args["workers"])

        class_per_tasks.append(args["num_classes"][ses])
        logit_init_ind += args["num_classes"][ses]
        logit_init_inds.append(logit_init_ind)

        trainloaders.append(train_loader)
        testloaders.append(val_loader)

    main_learner = Learner(model=model, args=args, sessions=sessions, trainloaders=trainloaders,
                           testloaders=testloaders, class_per_tasks=class_per_tasks, logit_init_inds=logit_init_inds,
                           use_cuda=args["use_cuda"], train_paths=train_paths)

    # train the base network on the defined tasks in parallel
    main_learner.learn()

    for ses in sessions:
        cfmat = main_learner.get_confusion_matrix(ses)
        np.save(args["checkpoint"] + "/confusion_matrix_" + str(ses) + "_" + str(args["test_case"]) + ".npy", cfmat)

        print('done with session {:d}'.format(ses))
        print('#################################################################################')
        while (1):
            if (is_all_done(ses, args["num_epochs"], args["checkpoint"])):
                break
            else:
                time.sleep(10)


json_file = '../../scripts/test_case0_5tasks_parallel.json'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', default=json_file, type=str, help='input json file')
    parsed = parser.parse_args()
    with open(parsed.json_file) as f:
        args = json.load(f)
    main(args)