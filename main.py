import argparse
from train import train
from evaluate import evaluate
import random
import torch

def main(args):
    logs_root = args.log_dir
    if args.mode == 'train':
        for seed in args.seeds:
            # 设置不同的随机种子,这样可以保证训练全局是固定的吗？
            print(f'Running with :\n seed {seed}\n mode {args.mode}\n data {args.data_name}\n normalization {args.normalization_type}\n optimizer {args.optimizer}\n lr {args.learning_rate}\n momentum {args.momentum}\n loss {args.loss}')
            
            random.seed(seed)
            torch.manual_seed(seed) # 这一步确定的，看看al 的 别人还有些其他写法
            args.log_dir = f'{logs_root}/{args.model_type}_{args.data_name}_{args.normalization_type}_{args.optimizer}_{args.learning_rate}_{args.momentum}_{args.loss}_{seed}'
            train(args)
            
    elif args.mode == 'test':
        for seed in args.seeds:    
            print(f'Running with :\n seed {seed}\n mode {args.mode}\n data {args.data_name}\n normalization {args.normalization_type}\n optimizer {args.optimizer}\n lr {args.learning_rate}\n momentum {args.momentum}\n loss {args.loss}')
            random.seed(seed)
            torch.manual_seed(seed) # 这一步确定的，看看al 的 别人还有些其他写法

            args.log_dir = f'{logs_root}/{args.model_type}_{args.data_name}_{args.normalization_type}_{args.optimizer}_{args.learning_rate}_{args.momentum}_{args.loss}_{seed}'
            evaluate(args)
    else:
        print("Invalid mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train',choices=['train', 'test'], help="Choose 'train' or 'test' mode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--model_type", type=str, default="neural_network", help="Model type (softmax or neural_network)")

    parser.add_argument("--optimizer", type=str, default="SGD", help="Model optimizer (softmax or neural_network)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--loss", type=str, default="CrossEntropyLoss", help="Model loss (softmax or neural_network)")
    
    parser.add_argument("--data_name", type=str, default="minst", help="data name (minst or cifar10)")
    parser.add_argument("--data_dir", type=str, default="/hone/shiyinglocal/Data", help="data dir")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 10], help="random seed")
    parser.add_argument("--normalization_type", type=str, default='standardize', help="regularization")
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--validation_split', type=int, default=0.2)
    args = parser.parse_args()
    main(args)
