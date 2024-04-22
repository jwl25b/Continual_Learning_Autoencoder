import random
import torch
import torchvision
from torchvision import datasets


class SplittedAndShuffledMNIST(datasets.MNIST):

    def __init__(self, root="./MNIST/", train=True, shuffle_idx=None, class_indices=None, task_num=None):
        super(SplittedAndShuffledMNIST, self).__init__(root, train, download=True)
        assert len(class_indices) == 2
        self.task_num = task_num
        if self.train:
            raw_mnist_train = datasets.MNIST(root=root, train=True, download=True)
            indices = [i for i, (e, c) in enumerate(raw_mnist_train) if c in class_indices]
            
            subset = torch.utils.data.Subset(raw_mnist_train.data, indices)
            self.data_train = torch.stack([torch.index_select(img.float().view(-1), 0, torch.tensor(shuffle_idx))/255 for img in subset])
            self.targets = torch.utils.data.Subset(raw_mnist_train.targets, indices)
            self.targets = torch.tensor([0 if int(x.item())==class_indices[0] else 1 for x in self.targets])
        else:
            raw_mnist_test = datasets.MNIST(root=root, train=False, download=True)
            indices = [i for i, (e, c) in enumerate(raw_mnist_test) if c in class_indices]
            
            subset = torch.utils.data.Subset(raw_mnist_test.data, indices)
            self.data_test = torch.stack([torch.index_select(img.float().view(-1), 0, torch.tensor(shuffle_idx))/255 for img in subset])
            self.targets = torch.utils.data.Subset(raw_mnist_test.targets, indices)
            self.targets = torch.tensor([0 if int(x.item())==class_indices[0] else 1 for x in self.targets])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        if self.train:
            img, target = self.data_train[index], self.targets[index]
        else:
            img, target = self.data_test[index], self.targets[index]

        return img, target, torch.tensor(self.task_num)

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data_train[sample_idx]]