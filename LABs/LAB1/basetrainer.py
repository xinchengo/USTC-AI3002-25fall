import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="mae"):
        """
        Initialize Trainer for linear or logistic regression.

        Args:
            model: LinearModel instance (LinearRegressionModel or LogisticRegressionModel).
            train_dataloader: DataLoader for training data.
            eval_dataloader: DataLoader for evaluation data (optional).
            save_dir: Directory to save checkpoints and plots (optional).
            learning_rate: Learning rate, default 0.01.
            eval_strategy: 'epoch' or 'step', when to evaluate.
            eval_steps: Evaluate every eval_steps if strategy is 'step'.
            num_epochs: Number of training epochs.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.num_epochs = num_epochs
        self.total_steps = num_epochs * len(self.train_dataloader)
        self.cur_step = 0
        self.eval_metric = eval_metric

    def compute_loss(self, batch_pred, batch_grd):
        """
        Compute loss based on model type.

        Args:
            batch_pred: Predicted values, shape [batch_size, out_features].
            batch_grd: True values/labels, shape [batch_size, out_features].

        Returns:
            float: Mean loss for the batch.
        """
        raise NotImplementedError("Subclasses must implement the backpropagation method.")

    def learning_rate_scheduler(self):
        pass

    def training_process(self, num_epochs=None):
        """
        Train the model for specified epochs.

        Args:
            num_epochs: Number of training epochs (overrides init if provided).

        Returns:
            tuple: (train_losses, eval_losses), lists of average losses per epoch.
        """
        num_epochs = num_epochs or self.num_epochs
        train_losses = []
        eval_losses = []
        self.cur_step = 0
        total_loss = 0.
        num_batches = 0
        
        print("*" * 10 + f" Start training, total epochs: {num_epochs}, total steps: {self.total_steps} " + "*" * 10)
        progress_bar = tqdm(total=self.total_steps, desc="Training")
        
        for epoch in range(num_epochs):
            for batch_x, batch_y in self.train_dataloader:
                self.cur_step += 1
                batch_y = batch_y.reshape(-1, 1)
                batch_y_pred = self.model.forward(batch_x).reshape(-1, 1)
                loss = self.compute_loss(batch_y_pred, batch_y)
                self.model.backpropagation(batch_x, batch_y, batch_y_pred, self.learning_rate)
                total_loss += loss
                num_batches += 1
                
                self.learning_rate_scheduler()

                postfix = {'train_loss': f'{loss:.4f}', 'cur_lr': f'{self.learning_rate:.5f}'}
                
                if self.eval_strategy == "step" and self.cur_step % self.eval_steps == 0:
                    eval_loss = self.eval()
                    eval_losses.append(eval_loss)
                    avg_loss = total_loss / num_batches
                    train_losses.append(avg_loss)
                    total_loss = 0.
                    num_batches = 0
                
                progress_bar.set_postfix(postfix)
                progress_bar.update(1)
            
            if self.eval_strategy == "epoch":
                avg_loss = total_loss / num_batches
                total_loss = 0.
                num_batches = 0
                train_losses.append(avg_loss)
                eval_loss = self.eval()
                eval_losses.append(eval_loss)
        
        return train_losses, eval_losses
    
    def train(self, num_epochs=None):
        train_losses, eval_losses = self.training_process(num_epochs=num_epochs)
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.save_dir, "model.npz")
            self.model.save_checkpoint(checkpoint_path)
            print(f"Successfully saved model to {checkpoint_path}")
        
        if self.save_dir and train_losses:
            os.makedirs(self.save_dir, exist_ok=True)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            if self.eval_strategy == "epoch":
                plt.xlabel('Epoch')
            else:
                plt.xlabel(f'Steps (per {self.eval_steps})')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(self.save_dir, 'train_loss_curve.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Training loss curve saved to {plot_path}")

            plt.plot(range(1, len(eval_losses) + 1), eval_losses, label=f'Eval {self.eval_metric}')
            if self.eval_strategy == "epoch":
                plt.xlabel('Epoch')
            else:
                plt.xlabel(f'Steps (per {self.eval_steps})')
            plt.ylabel('Loss')
            plt.title(f'Evaluation {self.eval_metric}')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(self.save_dir, 'eval_loss_curve.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Eval loss curve saved to {plot_path}")

        print("*" * 10 + f" Finish training! " + "*" * 10)

    def eval(self):
        """
        Evaluate the model on eval_dataloader.
        """
        if not self.eval_dataloader:
            return 0.

        total_grd = []
        total_pred = []
        
        for batch_x, batch_y in self.eval_dataloader:
            batch_y = batch_y.reshape(-1, 1)
            preds = self.model.forward(batch_x).reshape(-1, 1)
            total_grd.append(batch_y)
            total_pred.append(preds)
        
        total_pred = np.concatenate(total_pred, axis=0)
        total_grd = np.concatenate(total_grd, axis=0)
        
        return self.compute_metric(total_grd, total_pred)
    
    def compute_metric(self, all_grd, all_pred, metric = None):
        metric = metric or self.eval_metric
        if metric == "mae":
            return mean_absolute_error(all_grd, all_pred)
        if metric == "mse":
            return mean_squared_error(all_grd, all_pred)
        if metric == "r2":
            return r2_score(all_grd, all_pred)
        if metric == "acc":
            all_pred = (all_pred > 0.5).astype(int)
            return accuracy_score(all_grd, all_pred)
        if metric == "f1":
            all_pred = (all_pred > 0.5).astype(int)
            return f1_score(all_grd, all_pred)
        if metric == "auc":
            return roc_auc_score(all_grd, all_pred)

        
