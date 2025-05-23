<<<<<<< HEAD
import os
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import logging

# Local imports
from antigen_antibody_emb import configuration, antibody_antigen_dataset
from antibinder_model import antibinder, AntiModelIinitial
from utils.utils import CSVLogger_my

# Configure matplotlib for headless operation
matplotlib.use('agg')

# Suppress warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(__file__)

def setup_logging(log_dir, model_name):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_training_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TrainingMonitor:
    """Enhanced training monitor with visualization and tracking"""
    
    def __init__(self, log_dir, model_name):
        self.log_dir = log_dir
        self.model_name = model_name
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_precision': [],
            'train_f1': [],
            'train_recall': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Create plots directory
        self.plots_dir = os.path.join(log_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def update(self, epoch, metrics_dict):
        """Update metrics history"""
        for key, value in metrics_dict.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def plot_metrics(self):
        """Create and save training plots"""
        if len(self.metrics_history['epoch']) == 0:
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name} Training Metrics', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(self.metrics_history['epoch'], self.metrics_history['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.metrics_history['epoch'], self.metrics_history['train_acc'])
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[0, 2].plot(self.metrics_history['epoch'], self.metrics_history['train_f1'])
        axes[0, 2].set_title('Training F1 Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].grid(True)
        
        # Precision plot
        axes[1, 0].plot(self.metrics_history['epoch'], self.metrics_history['train_precision'])
        axes[1, 0].set_title('Training Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(True)
        
        # Recall plot
        axes[1, 1].plot(self.metrics_history['epoch'], self.metrics_history['train_recall'])
        axes[1, 1].set_title('Training Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].grid(True)
        
        # Learning rate plot
        if self.metrics_history['learning_rate']:
            axes[1, 2].plot(self.metrics_history['epoch'], self.metrics_history['learning_rate'])
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f'{self.model_name}_training_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def save_metrics_json(self):
        """Save metrics as JSON for later analysis"""
        json_path = os.path.join(self.log_dir, f'{self.model_name}_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        return json_path

class Trainer:
    def __init__(self, model, train_dataloader, args, logger, csv_logger, monitor, load=False):
        self.model = model
        self.train_dataloader = train_dataloader
        self.args = args
        self.logger = logger
        self.csv_logger = csv_logger
        self.monitor = monitor
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.load = load
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0
        
        # Move model to device
        self.model.to(self.device)
        
        if not self.load:
            self.init()
        else:
            self.logger.info("Loading existing model weights")

    def init(self):
        init = AntiModelIinitial()
        self.model.apply(init._init_weights)
        self.logger.info("Model initialized successfully!")

    def matrix_val(self, yhat, y):
        return accuracy_score(y, yhat), precision_score(y, yhat), f1_score(y, yhat), recall_score(y, yhat)

    def train(self, criterion, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        self.logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        self.logger.info(f"Batch size: {self.args.batch_size}")
        self.logger.info(f"Learning rate: {self.args.lr}")
        
        for epoch in range(self.start_epoch, epochs):
            epoch_start_time = time.time()
            self.model.train(True)
            train_loss = 0
            num_train = 0
            Y_hat = []
            Y = []
            
            # Progress bar for batches
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (antibody_set, antigen_set, label) in enumerate(pbar):
                # Move data to device
                antibody_set = [t.to(self.device) for t in antibody_set]
                antigen_set = [t.to(self.device) for t in antigen_set]
                label = label.to(self.device).float()

                # Forward pass
                probs = self.model(antibody_set, antigen_set)
                
                # Predictions and loss
                yhat = (probs > 0.5).long()
                loss = criterion(probs.view(-1), label.view(-1))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()
                num_train += antibody_set[0].shape[0]
                Y_hat.extend(yhat.cpu())
                Y.extend(label.cpu())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Batch': f'{batch_idx+1}/{len(self.train_dataloader)}'
                })

            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            Y_hat_concat = torch.cat([temp.view(-1) for temp in Y_hat], dim=0).numpy()
            Y_concat = torch.cat([temp.view(-1) for temp in Y], dim=0).numpy()
            
            train_acc, train_precision, train_f1, train_recall = self.matrix_val(Y_hat_concat, Y_concat)
            train_loss = train_loss / len(self.train_dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update learning rate scheduler
            scheduler.step(train_loss)

            # Log metrics to CSV (original system)
            self.csv_logger.log([epoch+1, train_loss, train_acc, train_precision, train_f1, train_recall])
            
            # Enhanced logging and monitoring
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_precision': train_precision,
                'train_f1': train_f1,
                'train_recall': train_recall,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            
            # Update monitoring
            self.monitor.update(epoch + 1, metrics)
            
            # Enhanced logging
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                f"F1: {train_f1:.4f}, Precision: {train_precision:.4f}, "
                f"Recall: {train_recall:.4f}, LR: {current_lr:.2e}, "
                f"Time: {epoch_time:.2f}s"
            )

            # Save best models (both loss and F1)
            if train_loss < self.best_loss:
                self.logger.info(f'New best loss: {train_loss:.4f} (prev: {self.best_loss:.4f})')
                self.best_loss = train_loss
                self.save_model(suffix='best_loss')
                
            if train_f1 > self.best_f1:
                self.logger.info(f'New best F1: {train_f1:.4f} (prev: {self.best_f1:.4f})')
                self.best_f1 = train_f1
                self.save_model(suffix='best_f1')

            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch + 1, optimizer, scheduler)
                
            # Generate plots every 25 epochs
            if (epoch + 1) % 25 == 0:
                plot_path = self.monitor.plot_metrics()
                if plot_path:
                    self.logger.info(f"Training plots updated: {plot_path}")

        # Final saves
        self.save_model(suffix='final')
        self.save_checkpoint(epochs, optimizer, scheduler)
        plot_path = self.monitor.plot_metrics()
        json_path = self.monitor.save_metrics_json()
        
        self.logger.info(f"Training completed!")
        if plot_path:
            self.logger.info(f"Final plots saved: {plot_path}")
        self.logger.info(f"Metrics saved: {json_path}")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        self.logger.info(f"Best F1: {self.best_f1:.4f}")

    def save_model(self, suffix='best'):
        """Save model with enhanced metadata"""
        ckpt_dir = os.path.join(PROJECT_ROOT, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(ckpt_dir, f"{self.args.model_name}_{self.args.latent_dim}_{suffix}.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Save detailed model (backward compatibility)
        detailed_save_path = os.path.join(
            ckpt_dir, 
            f"{self.args.model_name}_{self.args.data}_{self.args.batch_size}_{self.args.epochs}_{self.args.latent_dim}_{self.args.lr}.pth"
        )
        torch.save(self.model.state_dict(), detailed_save_path)
        
        # Save metadata
        metadata = {
            'model_name': self.args.model_name,
            'latent_dim': self.args.latent_dim,
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.lr,
            'epochs': self.args.epochs,
            'best_loss': self.best_loss,
            'best_f1': self.best_f1,
            'timestamp': datetime.now().isoformat(),
            'suffix': suffix
        }
        
        metadata_path = os.path.join(ckpt_dir, f"{self.args.model_name}_{self.args.latent_dim}_{suffix}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Model saved: {model_path}")

    def save_checkpoint(self, epoch, optimizer, scheduler):
        """Save full training checkpoint"""
        ckpt_dir = os.path.join(PROJECT_ROOT, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_f1': self.best_f1,
            'args': vars(self.args),
            'metrics_history': self.monitor.metrics_history
        }
        
        checkpoint_path = os.path.join(ckpt_dir, f"{self.args.model_name}_checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, optimizer, scheduler):
        """Load training checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.best_f1 = checkpoint['best_f1']
            self.start_epoch = checkpoint['epoch']
            if 'metrics_history' in checkpoint:
                self.monitor.metrics_history = checkpoint['metrics_history']
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AntiBinder Model Training")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=36, help='Latent dimension size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=6e-5, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='AntiBinder', help='Model name')
    parser.add_argument('--device', type=str, default='0', help="GPU device(s) to use")
    parser.add_argument('--data', type=str, default='train', help="Dataset split to use")
    parser.add_argument('--load_model', action='store_true', help='Load existing model weights')
    parser.add_argument('--resume_checkpoint', type=str, default='', help='Path to checkpoint to resume from')
    
    args = parser.parse_args()

    # Set CUDA device visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup enhanced logging
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    logger = setup_logging(log_dir, args.model_name)
    monitor = TrainingMonitor(log_dir, args.model_name)
    
    logger.info(f"Using device: {device} (CUDA_VISIBLE_DEVICES={args.device})")
    logger.info(f"Arguments: {vars(args)}")

    # Configure model components
    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings', 1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings', 149)

    # Initialize model
    model = antibinder(
        antibody_hidden_dim=1024,
        antigen_hidden_dim=1024,
        latent_dim=args.latent_dim,
        res=False
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup dataset
    if args.data == 'train':
        data_path = os.path.join(PROJECT_ROOT, 'datasets', 'combined_training_data_split.csv')
    else:
        raise ValueError(f"Unsupported data argument: {args.data}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    train_dataset = antibody_antigen_dataset(
        antigen_config=antigen_config,
        antibody_config=antibody_config,
        data_path=data_path,
        train=True,
        test=False,
        rate1=0.8
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")

    # Setup CSV logging (keeping original system)
    log_path = os.path.join(
        log_dir,
        f"{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv"
    )
    csv_logger = CSVLogger_my(['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall'], log_path)

    # Load model if requested
    if args.load_model:
        load_path = os.path.join(PROJECT_ROOT, "ckpts", f"{args.model_name}_{args.latent_dim}_best_loss.pth")
        if not os.path.exists(load_path):
            load_path = os.path.join(PROJECT_ROOT, "ckpts", f"{args.model_name}_{args.latent_dim}_best.pth")
        
        if os.path.exists(load_path):
            model.load_state_dict(torch.load(load_path, map_location=device))
            logger.info(f"Loaded model weights from {load_path}")
        else:
            logger.warning(f"Model weight file not found. Starting from scratch.")
            args.load_model = False

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        logger=logger,
        csv_logger=csv_logger,
        monitor=monitor,
        args=args,
        load=args.load_model
    )

    # Resume from checkpoint if specified
    if args.resume_checkpoint:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        if trainer.load_checkpoint(args.resume_checkpoint, optimizer, scheduler):
            logger.info(f"Resumed training from checkpoint: {args.resume_checkpoint}")

    # Start training
    criterion = nn.BCELoss()
    logger.info(f"Starting training for {args.epochs} epochs...")
    trainer.train(criterion=criterion, epochs=args.epochs)
    logger.info("Training completed!")
=======
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
>>>>>>> 1a30f7b606d441657fc7b554357c70f7fa59651e
