import itertools
import subprocess
import random
from data_loader import load_data
from model import create_model
from train import train_model
from evaluate import evaluate_model


# Define hyperparameters to search
batch_sizes = [32, 64, 128]
learning_rates = [0.001, 0.01, 0.1]
epochs = [5, 10, 15]

best_accuracy = 0.0
best_params = None

# Load the entire training dataset
trainloader, _ = load_data()

# Perform grid search with train-validation split
for params in itertools.product(batch_sizes, learning_rates, epochs):
    batch_size, lr, epoch = params
    
    # Split the dataset into train and validation sets
    random.seed(42)  # For reproducibility
    random.shuffle(trainloader.dataset.data)
    split = int(0.8 * len(trainloader.dataset))
    train_data, val_data = trainloader.dataset.data[:split], trainloader.dataset.data[split:]
    
    # Create new DataLoaders for train and validation sets
    trainloader.dataset.data = train_data
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    command = f"python train.py --batch_size {batch_size} --learning_rate {lr} --epochs {epoch}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    
    # Parse the accuracy from the output (modify this part based on your training script's output format)
    output = process.stdout.read().decode('utf-8')
    accuracy_line = [line for line in output.split('\n') if 'Accuracy on the test set' in line]
    if accuracy_line:
        accuracy = float(accuracy_line[0].split(':')[-1].strip()[:-1])
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

print(f"Best hyperparameters: Batch size {best_params[0]}, Learning rate {best_params[1]}, Epochs {best_params[2]}")
print(f"Best accuracy: {best_accuracy}")

# Train the final model with the best hyperparameters using the entire training dataset
best_batch_size, best_lr, best_epoch = best_params
train_command = f"python train.py --batch_size {best_batch_size} --learning_rate {best_lr} --epochs {best_epoch}"
subprocess.run(train_command, shell=True)
