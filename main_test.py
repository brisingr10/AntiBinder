import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from antigen_antibody_emb import * 
from antibinder_model import *
import torch
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader 
from copy import deepcopy 
from tqdm import tqdm
import os
import sys
import argparse
from utils.utils import CSVLogger_my
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score,roc_auc_score,confusion_matrix
sys.path.append ('../') 
import warnings 
warnings.filterwarnings("ignore")


# Define Project Root
PROJECT_ROOT = os.path.dirname(__file__) 

class Trainer():
    def __init__(self,model,valid_dataloader,args,logger) -> None:
        self.model = model
        self.valid_dataloader = valid_dataloader
        self.args = args
        self.logger = logger
        # self.grad_clip = args.grad_clip # cLip gradients at this value, or disable if == 0.0
        self.best_loss = None

    def matrix(self,yhat,y):
        return sum (y==yhat)
    

    def matrix_val(self,yhat,y,yscores) :
        print(sum(yhat))

        TN,FP,FN,TP =0,0,0,0
        cm = confusion_matrix(y, yhat).ravel()
        # print (cm)
        if len(cm)== 1:
            # print (y[0].item(), yhat[0].item())
            if y[0].item() == yhat[0].item()== 0:
                TN= cm[0]
            elif y[0].item() == yhat[0].item() == 1:
                TP = cm[0]
        else:
            TN,FP,FN,TP = confusion_matrix(y,yhat).ravel()
        if len(np.unique(y))>1:
            roc_auc = roc_auc_score(y, yscores)
        else:
            roc_auc = None
        
        return roc_auc, precision_score(y, yhat),accuracy_score(y,yhat), recall_score(y, yhat),f1_score(y,yhat),TN,FP,FN,TP
    

    def valid(self):
        self.model.eval()
        val_acc = 0
        Y_hat = []
        Y = []
        Y_scores = []
        with torch.no_grad():
            for antibody_set, antigen_set, label in tqdm(self.valid_dataloader):
                probs = self.model(antibody_set, antigen_set)
                # print(probs)
                #10*2
                y = label.float()
                yhat = (probs>0.5).long().cuda()
                y_scores = probs

                Y_hat.extend(yhat)
                Y.extend(y)
                Y_scores.extend(y_scores)

        auc, val_prescision, val_acc, recall, val_f1, TN, FP, FN, TP= self.matrix_val((torch.cat([temp.view(1, -1) for temp in Y_hat], dim=0)).long().cpu().numpy(),
                                                                                      torch.tensor(Y),
                                                                                      (torch.cat([temp2.view(1, -1) for temp2 in Y_scores], dim=0)).cpu().numpy())
        return auc, val_prescision, val_acc, recall, val_f1, TN, FP, FN, TP
    

    def train(self):
        val_auc, val_prescision, val_acc, val_recall, val_f1, TN, FP, FN, TP =self.valid()
        self.logger.log([val_auc, val_prescision, val_acc, val_recall, val_f1, TN, FP, FN, TP])

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=36)
    parser.add_argument('--model_name', type=str, default= 'AntiBinder')
    # parser.add_argument('--device', type=str, default='0,1')
    parser.add_argument('--data', type=str, default= 'test')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings', 1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings',149)
 


    model = antibinder(antibody_hidden_dim=1024,antigen_hidden_dim=1024,latent_dim=args.latent_dim,res=False).cuda()
    print(model)
   
    # load model
    # Construct relative path for checkpoints (assuming a 'ckpts' directory)
    ckpt_dir = os.path.join(PROJECT_ROOT, "ckpts") 
    # Load the generic 'best' model saved by the trainer
    model_path = os.path.join(ckpt_dir, f"{args.model_name}_{args.latent_dim}_best.pth") # This now matches the trainer's best model save path
    weight = torch.load(model_path) 
    model.load_state_dict(weight)
    print("load success")


    # choose test dataset
    if args.data == 'test':
        # Construct relative data path
        data_path = os.path.join(PROJECT_ROOT, 'datasets', 'combined_training_data.csv') # Replace with your actual test data file
  

    print (data_path)
    val_dataset =antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config,data_path=data_path, train=False, test=True, rate1=0)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

    # Construct relative path for logs
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True) # Ensure directory exists
    log_path = os.path.join(log_dir, f"{args.model_name}_{args.latent_dim}_{args.data}.csv")
    logger = CSVLogger_my(['val_auc', 'val_prescision', 'val_acc', 'val_recall', 'val_f1', 'TN', 'FP', 'FN', 'TP'], log_path)
    
    scheduler = None
    trainer = Trainer(
        model = model,
        valid_dataloader = val_dataloader,
        logger = logger,
        args= args,
        )
    
    trainer.train()
