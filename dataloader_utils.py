import random
import numpy.random as np_rand
import torch.utils.data as data

from permutedMNIST import PermutedMNIST

def get_conbined_permute_mnist(num_task, batch_size, random_seed):
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