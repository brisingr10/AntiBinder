import os, sys
# Remove initial argparse setup
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='0', help="which GPU to use")
# # … add the rest of your args …
# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device # Move this down after args are parsed

# Keep this if needed for relative imports from parent dir
# sys.path.append('../')   

from antigen_antibody_emb import * 
from antibinder_model import *
import matplotlib
matplotlib.use('agg')    # choose 'agg' (a headless, file‑output backend)
import torch
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader 
from copy import deepcopy 
from tqdm import tqdm
# import sys # Duplicate import
import argparse # Keep this one
from utils.utils import CSVLogger_my
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score
# sys.path.append ('../') # Duplicate sys.path append
import warnings 
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device) # Print after setting CUDA_VISIBLE_DEVICES

PROJECT_ROOT = os.path.dirname(__file__) 

class Trainer():
    # ... (Trainer class definition remains the same) ...
    def __init__(self,model,train_dataloader,args,logger,load=False) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        # self.vaLid_dataloader = valid_dataloader
        # self.test_dataloader = test_dataloader
        self.args = args
        self.logger = logger
        # self.grad_clip = args.grad_clip # cLip gradients at this value, or disable if == 0.0
        self.best_loss = None
        self.load = load

        if self.load==False:
            self.init()
        else:
            print("no init model")

    def init(self):
        init = AntiModelIinitial()
        self.model.apply(init._init_weights)
        print("init successfully!")


    def matrix(self,yhat,y):
        return sum (y==yhat)
    

    def matrix_val(self,yhat,y) :
        # print(sum(yhat))
        return accuracy_score(y,yhat), precision_score(y, yhat), f1_score(y,yhat), recall_score(y, yhat)
    

    def train(self, criterion, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
        for epoch in range(epochs) :
            self.model.train(True)
            train_acc = 0
            train_loss = 0
            num_train = 0
            Y_hat = []
            Y = []
            for antibody_set, antigen_set, label in tqdm(self.train_dataloader):
                antibody_set = [t.to(device) for t in antibody_set]
                antigen_set  = [t.to(device) for t in antigen_set]
                label        = label.to(device).float()

                probs = self.model(antibody_set, antigen_set)

                yhat = (probs>0.5).long()
                y = label.float().cuda()
                loss = criterion(probs.view(-1),y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_train += antibody_set[0].shape[0]
                Y_hat.extend(yhat)
                Y.extend(y)

            train_acc, train_precision, train_f1, recall = self.matrix_val((torch.cat([temp.view(1, -1) for temp in Y_hat], dim=0)).long().cpu().numpy(),
                                                                            torch.tensor(Y))
            train_loss = train_loss / num_train
            train_loss = np.exp(train_loss)

            self.logger.log([epoch+1, train_loss, train_acc, train_precision,train_f1,recall])

            if self.best_loss==None or train_loss < self.best_loss:
                print('epoch: ',epoch, 'saving...')
                self.best_loss = train_loss
                self.save_model()


    def save_model(self):
            # Construct relative path for checkpoints
            ckpt_dir = os.path.join(PROJECT_ROOT, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True) # Ensure directory exists
            
            # Save model with detailed parameters in the filename
            detailed_save_path = os.path.join(ckpt_dir, f"{self.args.model_name}_{self.args.data}_{self.args.batch_size}_{self.args.epochs}_{self.args.latent_dim}_{self.args.lr}.pth")
            torch.save(self.model.state_dict(), detailed_save_path)
            print(f"Model saved to {detailed_save_path}")

            # Also save/overwrite a generic 'best' model file
            best_save_path = os.path.join(ckpt_dir, f"{self.args.model_name}_{self.args.latent_dim}_best.pth")
            torch.save(self.model.state_dict(), best_save_path)
            print(f"Best model updated at {best_save_path}")


if __name__ == "__main__":
    # Move parser definition and all argument additions here
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--latent_dim', type=int, default=36)
    # In certain datasets, an early stopping strategy is required to achieve optimal results.
    parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay used in optimizer') # 1e-5
    parser.add_argument('--lr', type=float, default=6e-5, help='learning rate')
    parser.add_argument('--model_name', type=str, default= 'AntiBinder')
    # parser.add_argument('--cuda', type=bool, default=True) # Use torch.cuda.is_available() instead
    parser.add_argument('--device', type=str, default='0', help="which GPU to use (e.g., '0' or '0,1')")
    parser.add_argument('--data', type=str, default='train', help="Dataset split to use (e.g., 'train')")
    
    # Parse arguments *after* defining them all
    args = parser.parse_args() 

    # Set CUDA device visibility *after* parsing args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={args.device})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings',1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings',149)

    # Use the global 'device' variable determined earlier
    model = antibinder(antibody_hidden_dim=1024,antigen_hidden_dim=1024,latent_dim=args.latent_dim,res=False).to(device) 
    print(model)

    # # muti-gpus - Consider using DistributedDataParallel for multi-GPU training if needed
    # if len(args.device.split(',')) > 1 and torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model) 
    # model.to(device) # Ensure model is on the correct device(s)


    # here choose dataset
    if args.data == 'train':
        # Construct relative path for dataset
        data_path = os.path.join(PROJECT_ROOT, 'datasets', 'combined_training_data_split.csv')
    # elif args.data == 'train_2':
    #     data_path = '' # Define path similarly if needed
    else:
        # Handle cases where args.data is not 'train' if necessary
        raise ValueError(f"Unsupported data argument: {args.data}")


    # print (data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Combined dataset not found at: {data_path}. Please run combine_data.py and heavy_chain_split.py first.")
        
    # Use rate1=1 to use the whole dataset for training as split by the dataset class itself
    train_dataset = antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config,data_path=data_path, train=True, test=False, rate1=0.8) # Assuming 80/20 split logic in dataset class
    # ... existing validation dataset setup (commented out) ...

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size) # Shuffle training data
    # ... existing validation dataloader setup (commented out) ...
  
    # Construct relative path for logs
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True) # Ensure directory exists
    log_path = os.path.join(log_dir, f"{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv")
    logger = CSVLogger_my(['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall'], log_path)
    
    scheduler = None

    # load model if needs
    load = False
    if load:
        # Define path to load model from, e.g., the 'best' model
        load_path = os.path.join(PROJECT_ROOT, "ckpts", f"{args.model_name}_{args.latent_dim}_best.pth")
        if os.path.exists(load_path):
            weight = torch.load(load_path, map_location=device) # Ensure loading to correct device
            model.load_state_dict(weight)
            print(f"Loaded model weights from {load_path}")
        else:
             print(f"Warning: Model weight file not found at {load_path}. Starting training from scratch.")
             load = False # Set load back to False if file not found


    trainer = Trainer(model=model,
        train_dataloader=train_dataloader,
        # valid_dataloader=val_dataLoader,
        # test_dataLoader=test._dataloader,
        logger = logger,
        args= args,
        load=load
        )

    criterion = nn.BCELoss()
    trainer.train(criterion=criterion, epochs=args.epochs)