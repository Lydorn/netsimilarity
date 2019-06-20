import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DoubleMnist:

    def __init__(self, dataset, batch_size, device):
        assert type(dataset) == datasets.MNIST and dataset.transform is not None
        assert type(batch_size) == int and batch_size > 0

        # save dataset information
        self.transform = dataset.transform
        self.images = dataset.data
        self.labels = dataset.targets
        self.batch_size = batch_size
        self.device = device
        self.n_batches_per_epoch = 60000 // self.batch_size

        # setup
        self.labels_list = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_list}

    def epoch_iterator(self):
        """Yield batches of data within one epoch."""
        for _ in range(self.n_batches_per_epoch):
            # gather and return the data for one batch element
            current_label = np.random.choice(10)
            images_1, labels_1 = self.get_data(current_label)
            images_2, labels_2 = self.get_data(current_label)

            # yield everything
            yield {
                "inputs": (images_1, images_2),
                "targets": (labels_1, labels_2),
                "length": self.n_batches_per_epoch,
                "_self": self,
            }

    def global_iterator(self):
        """Yield epoch iterators indefinitely."""
        while True:
            yield self.epoch_iterator()

    def get_data(self, current_label):
        """Gather and return the data for one batch element."""
        batch_images, batch_labels = [], []
        for _ in range(self.batch_size):
            index = np.random.choice(self.label_to_indices[current_label])
            image = Image.fromarray(self.images[index].numpy(), mode="L")
            batch_images.append(self.transform(image))
            batch_labels.append(self.labels[index])
        return (torch.stack(batch_images, dim=0).to(self.device),
                torch.stack(batch_labels, dim=0).to(self.device))


def get_mnist_iterators(data_path, batch_size, device):
    # transforms
    mean, std = 0.1307, 0.3081
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])

    # make the datasets
    train_dataset = DoubleMnist(
        datasets.MNIST(data_path, train=True, download=True, transform=transform),
        batch_size=batch_size, device=device)
    valid_dataset = DataLoader(
        datasets.MNIST(data_path, train=False, download=True, transform=transform),
        batch_size=1000, shuffle=True)
    return train_dataset.global_iterator(), valid_dataset
