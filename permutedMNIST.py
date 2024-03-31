import random
import torch
from torchvision import datasets


class PermutedMNIST(datasets.MNIST):

    def __init__(self, root="./MNIST/", train=True, permute_idx=None, task_num=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28
        self.task_num = task_num
        if self.train:
            self.data_train = torch.stack([img.float().view(-1)[permute_idx] / 255
                                           for img in self.data])
        else:
            self.data_test = torch.stack([img.float().view(-1)[permute_idx] / 255
                                          for img in self.data])

    def __getitem__(self, index):

        if self.train:
            img, target = self.data_train[index], self.targets[index]
        else:
            img, target = self.data_test[index], self.targets[index]

        return img, target, torch.tensor(self.task_num)

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data_train[sample_idx]]