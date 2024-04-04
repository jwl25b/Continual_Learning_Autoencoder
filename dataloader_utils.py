import random
import numpy.random as np_rand

import torch.utils.data as data
import torchvision

from permutedMNIST import PermutedMNIST
from splittedMNIST import SplittedMNIST
from permutedAndSplittedMNIST import PermutedAndSplittedMNIST

def get_conbined_permute_mnist(num_task, batch_size, random_seed):
    assert num_task>0
    train_datasets = {}
    test_datasets = {}
    np_rand.seed(random_seed)
    idx = list(range(28 * 28))
    for i in range(num_task):
        train_datasets[i] = PermutedMNIST(train=True, permute_idx=idx, task_num=i)
        test_datasets[i] = PermutedMNIST(train=False, permute_idx=idx, task_num=i)
        np_rand.shuffle(idx)

    train_dataset = data.ConcatDataset([x for _,x in train_datasets.items()])
    test_dataset = data.ConcatDataset([x for _,x in test_datasets.items()])

    train_dataloader = data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle = False,
                                                  num_workers = 4)
    test_dataloader = data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle = False)


    return random.sample(list(train_dataloader), len(train_dataloader)), random.sample(list(test_dataloader), len(test_dataloader))

def check_instance(values, items):
    for _, item in items:
        for value in values:
            if value in item:
                #print(value, item)
                return True
    return False

def get_conbined_split_mnist(num_task, batch_size, random_seed):
    assert num_task<=5 and num_task>0
    np_rand.seed(random_seed)
    classes = {}
    i=0
    while(i<num_task):
        current_class = np_rand.choice(range(10), 2, replace=False)
        if check_instance(current_class, classes.items()):
            continue
        else:
            classes[i]=(current_class)
            i+=1

    classes = [list(x) for _, x in classes.items()]
    print(f"split classes: {classes}")

    train_datasets = {}
    test_datasets = {}
    
    for j in range(num_task):
        class_indices = classes[j]
        train_datasets[j] = SplittedMNIST(train=True, class_indices=class_indices, task_num=j)
        test_datasets[j] = SplittedMNIST(train=False, class_indices=class_indices, task_num=j)

    train_dataset = data.ConcatDataset([x for _,x in train_datasets.items()])
    test_dataset = data.ConcatDataset([x for _,x in test_datasets.items()])

    train_dataloader = data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle = False,
                                                  num_workers = 4)
    test_dataloader = data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle = False)


    return random.sample(list(train_dataloader), len(train_dataloader)), random.sample(list(test_dataloader), len(test_dataloader))

def get_conbined_permute_and_split_mnist(num_task, batch_size, random_seed):
    assert num_task<=5 and num_task>0
    np_rand.seed(random_seed)
    classes = {}
    i=0
    while(i<num_task):
        current_class = np_rand.choice(range(10), 2, replace=False)
        if check_instance(current_class, classes.items()):
            continue
        else:
            classes[i]=(current_class)
            i+=1

    classes = [list(x) for _, x in classes.items()]
    print(f"split classes: {classes}")

    train_datasets = {}
    test_datasets = {}

    idx = list(range(28 * 28))
    np_rand.shuffle(idx)
    
    for j in range(num_task):
        class_indices = classes[j]
        train_datasets[j] = PermutedAndSplittedMNIST(train=True, permute_idx=idx, class_indices=class_indices, task_num=j)
        test_datasets[j] = PermutedAndSplittedMNIST(train=False, permute_idx=idx, class_indices=class_indices, task_num=j)

    train_dataset = data.ConcatDataset([x for _,x in train_datasets.items()])
    test_dataset = data.ConcatDataset([x for _,x in test_datasets.items()])

    train_dataloader = data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle = False,
                                                  num_workers = 4)
    test_dataloader = data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle = False)


    return random.sample(list(train_dataloader), len(train_dataloader)), random.sample(list(test_dataloader), len(test_dataloader))