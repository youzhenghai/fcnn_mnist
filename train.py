import torch
import torch.nn as nn
import torch.optim as optim
import data
from models import SoftmaxClassifier, NeuralNetworkClassifier
import os

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,val_loader, _ = data.get_mnist_data(args.batch_size,args.validation_split,args.normalization_type,args.data_name,args.data_dir)

    if args.model_type == 'softmax':
        model = SoftmaxClassifier()
    elif args.model_type == 'neural_network':
        model = NeuralNetworkClassifier()
    else:
        raise ValueError("Invalid model type. Supported values are 'softmax' and 'neural_network'.")
        

    model.to(device)


    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'NLLLoss':
        criterion = nn.NLLLoss()
    elif args.loss == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid loss. Supported values are 'CrossEntropyLoss' and 'NLLLoss'.")

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Invalid optimizer. Supported values are 'SGD' and 'Adam'.")


    best_accuracy = 0.0  # Track the best validation accuracy
    best_model_state = None  # Track the state of the best model

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / len(train_loader)}")

        # Evaluate the model on the validation set
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

        # Check if the current model is the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
            model_save_dir = os.path.join(args.log_dir)  # 模型保存路径 join ,两边都是路径吗
    
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # Save the best model state
    torch.save(best_model_state, f"{model_save_dir}/epoch_{str(epoch)}_model.pth")



