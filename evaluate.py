import torch
import data
from models import SoftmaxClassifier, NeuralNetworkClassifier
import os

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _, _,test_loader = data.get_mnist_data(args.batch_size)


    logs_path = args.log_dir
    file_list = os.listdir(logs_path)
    # 遍历文件列表
    for filename in file_list:
        if filename.endswith(".pth"):
            # 如果文件以 .pth 扩展名结尾，读入文件
            file_path = os.path.join(logs_path, filename)
            if args.model_type == 'softmax':
                model = SoftmaxClassifier()
            else:
                model = NeuralNetworkClassifier()

            model.load_state_dict(torch.load(file_path))
            model.to(device)
            model.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Accuracy on the test set: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'test'], help="Choose 'train' or 'test' mode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--model_type", type=str, default="softmax", help="Model type (softmax or neural_network)")
    args = parser.parse_args()
    evaluate(args)
