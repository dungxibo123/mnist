import os,sys
import argparse


import torch
import torch.nn.functional as F

from dataset import get_train_val_loader
from model import BasicConvolutionNeuralNetwork

from tqdm import tqdm
import mlflow
import logging
import warnings
import datetime # using ds_start macro in Airflow
#logger = logging.Logger()



def create_model(opt):
    model = BasicConvolutionNeuralNetwork(opt)
    return model
    #pass
    
def create_optimizer(opt):
    pass 

def get_opt():
    args = argparse.ArgumentParser()


    # Training Argument
    args.add_argument("-b", "--batch-size", type=int, default=32)
    args.add_argument("-e", "--epoch", type=int, default=100)
    args.add_argument("--weight-decay", type=float, default=0)
    args.add_argument("--optimizer", choices=["sgd", "adam", "adamw"])
    args.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    args.add_argument("-vb", "--val-batch-size", type=int, default=128)
    args.add_argument("--use-gpu", action="store_true")
    args.add_argument("--test-size", default=0.2, type=float)
    args.add_argument("--workers", default=1, type=int)
    args.add_argument("--label-smoothing", default=0.1, type=float)


    # Network setting
    args.add_argument("--channels", nargs="*", default=[20,10], type=int)
    args.add_argument("--augment", action='store_true')
    args.add_argument("--dropout", type=int, default=0)
    args.add_argument("--batch-norm", action='store_true')
    args.add_argument("--kernel-size", type=int, default=5)
    args.add_argument("--maxpool-kernel-size", type=int, default=2)
    args.add_argument("--dense-size", type=int, default=128)
    args.add_argument("--num-classes", type=int, default=10)

    # Variables setting
    args.add_argument("--checkpoint", action='store_true')
    args.add_argument("--checkpoint-path", type=str, default="./checkpoints")
    args.add_argument("--model-path", type=str, default="./model")
    args.add_argument("--data", choices=["MNIST","FMNIST"], required=True)
    args.add_argument("--data-path", type=str, required=True)

    args.add_argument("--unbalanced", action='store_true')


    # Network Argument
    opt = args.parse_args()
    return opt

def criterion():
    return F.binary_cross_entropy_with_logits
def evaluate(model, val_loader, opt):
    model.eval()
    correct = 0
    total = 0 
    running_loss = 0
    count = 0 
    with torch.no_grad():
        for batch_id , test_data in enumerate(val_loader,0):
            count += 1
            data, label = test_data
            label = F.one_hot(label, num_classes=opt.num_classes)
            if opt.use_gpu:
                data = data.to("cuda")
                label = label.to("cuda")
            outputs = model(data)
            _, correct_labels = torch.max(label, 1) 
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == correct_labels).sum().item()
            running_loss += criterion()(
                outputs.float(), label.float()).item()
    #        print(f"--> Total {total}\n-->batch_id: {batch_id + 1}")
    acc = round(correct/total * 1.0 , 5)
    running_loss /= count 
    return running_loss, acc
def train_one_iter(model, optim, train_load, val_loader, opt, epoch):
    losses = 0
    model.train()
    with tqdm(train_load,  unit="batch", position=0, leave=True) as tp:
        tp.set_description(f"Epoch {epoch}/{opt.epoch}")
        for (batch_x, batch_y) in tp:
             
            optim.zero_grad()
            batch_y = torch.nn.functional.one_hot(batch_y, num_classes = opt.num_classes)
            if opt.use_gpu:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")
            #print(batch_x.shape, batch_y.shape)
            outputs = model(batch_x)
            _, correct_labels = torch.max(batch_y, 1)
            _, predicted = torch.max(outputs.data, 1)
            total = batch_y.size(0)
            correct = (predicted == correct_labels).sum().item()
            train_acc = round(correct / total, 5)
            loss = criterion()(outputs.float(), batch_y.float())
            loss_item = loss.item()
            losses += loss_item
            loss.backward()
            optim.step()
            tp.set_postfix(loss=loss_item, train_acc=train_acc)
        val_loss, val_acc = evaluate(model, val_loader, opt)
        print(f"Epoch {epoch}/{opt.epoch} is finished with validation accuracy is {val_acc}")

    return model, optim, losses, val_loss, val_acc




def main():
    opt = get_opt()
    train_loader, val_loader = get_train_val_loader(opt)
    
    model = create_model(opt)
    if opt.use_gpu:
        model = model.to("cuda")

#    model.get_size_after_flatten()
#    trial_data = torch.rand((32,1,28,28))


    ### MLFLOW experiment find
    
    exps = mlflow.search_experiments(view_type=3, filter_string=f"name ILIKE \'{os.environ.get('MLFLOW_EXPERIMENT_NAME')}\'")
    if len(exps) < 1:
        print("Experiment could not be found. Init new one")
        experiment_id = mlflow.create_experiment(
            name=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
            
            tags={"version": "v1" , "priority": "P1"}
        )
        print(f"Experiment with id {os.environ.get('MLFLOW_EXPERIMENT_NAME')} had been initialized")

    else:
        print(exps)
        experiment_id = exps[0].experiment_id

    
#    print(model(trial_data))
    optim = None
    if opt.optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay, momentum=0.93, nesterov=True)
    elif opt.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay, eps=1e-7)
    elif opt.optimizer == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay, eps=1e-7)
    print(model)
    
    #for i in tqdm.tqdm(range(opt.epoch)):

    #with tqdm(range(1, opt.epoch + 1), position=0, leave=True) as tp:

    best_model = None
    best_val_acc = 0
    best_epoch = -1
    losses = []
    val_accuracies = []
    val_losses = []


    mlflow.start_run(
                        experiment_id=experiment_id
                    )
    mlflow.log_params(vars(opt))
    for i in range(1, opt.epoch+1): 

        
        model, optim, epoch_loss, val_loss, val_acc = train_one_iter(model, optim, train_loader, val_loader, opt, epoch = i)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        mlflow.log_metrics(
            {
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "val_acc": val_acc

            },
            step=i
        )
        if i % 5 == 0 and opt.checkpoint:
            torch.save({
                'opt': opt,
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
            }, opt.checkpoint_path + "/model.pt")
        if val_acc > best_val_acc:
            best_model = model
            best_val_acc = val_acc
            best_epoch = i
        #tp.set_postfix(loss=epoch_loss, val_loss=val_loss, val_acc=val_acc)

    mlflow.end_run()

    print("Running success! Evaluate here")

if __name__=="__main__":
    main()
