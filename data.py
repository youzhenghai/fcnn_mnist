import torchvision
import torchvision.transforms as transforms
import torch


def get_mnist_data(batch_size, validation_split=0.2, normalization_type='standardize', data_name='mnist',data_dir='/home/shiyinglocal/Data'):
    if normalization_type == 'standardize':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    elif normalization_type == 'min-max':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    else:
        raise ValueError("Invalid normalization_type. Supported values are 'standardize' and 'min-max'.")
    

    # if data_name == 'mnist':
    #     data_dir = '/home/shiyinglocal/Data/MNIST'  # 修改为正确的 MNIST 数据目录路径
    # elif data_name == 'cifar10':
    #     data_dir = '/home/shiyinglocal/Data/CIFAR10'  # 修改为正确的 CIFAR-10 数据目录路径
    # else:
    #     raise ValueError("Invalid data_name. Supported values are 'mnist' and 'cifar10'.")

    dataset = torchvision.datasets.MNIST(root="/home/shiyinglocal/Data", train=True, transform=transform, download=True)

    
    # Calculate the size of the validation set based on the validation_split
    num_samples = len(dataset)
    num_validation_samples = int(validation_split * num_samples) # 0 ?
    num_train_samples = num_samples - num_validation_samples
    
    # Split the dataset into training and validation sets
    # random need a seed ?
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train_samples, num_validation_samples])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_data = torchvision.datasets.MNIST(root='/home/shiyinglocal/Data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)    

    return train_loader, val_loader, test_loader


# def get_mnist_data(batch_size):
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#     train_data = torchvision.datasets.MNIST(root='/home/shiyinglocal/Data', train=True, transform=transform, download=True)
#     test_data = torchvision.datasets.MNIST(root='/home/shiyinglocal/Data', train=False, transform=transform, download=True)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader




# def get_mnist_data(batch_size, validation_split=0.2,normalization_type='min-max'):
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#     dataset = torchvision.datasets.MNIST(root='/home/shiyinglocal/Data', train=True, transform=transform, download=True)
    
#     # Calculate the size of the validation set based on the validation_split
#     num_samples = len(dataset)
#     num_validation_samples = int(validation_split * num_samples)
#     num_train_samples = num_samples - num_validation_samples
    
#     # Split the dataset into training and validation sets
#     # random need a seed ?
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train_samples, num_validation_samples])
    
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     test_data = torchvision.datasets.MNIST(root='/home/shiyinglocal/Data', train=False, transform=transform, download=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)    

#     return train_loader, val_loader ,test_loader
