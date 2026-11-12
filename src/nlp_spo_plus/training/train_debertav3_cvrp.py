import os
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from pathlib import Path
import pyepo
import time

# internal imports
from nlp_spo_plus.data.cvrp_text_dataset import cvrpTextDataset
from nlp_spo_plus.models.debertav3_cvrp import DeBERTaCVRP1
from nlp_spo_plus.optimization.cvrp import cvrpMTZModel, cvrpMTZModelRel

# dynamic resolution of project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# is cuda available
print(torch.cuda.is_available())

@dataclass
class TrainConfig:
    # configuration for the two stage dfl training pipeline
    epochs_mse: int = 15
    epochs_spo: int = 50
    batch_size: int = 8 
    lr_encoder: float = 2e-5  
    lr_head: float = 1e-4     
    num_processes: int = 4    
    log_interval: int = 25    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # metadata dynamically tracked for plotting and saving
    nlp_model_name: str = "BaseModel"
    exact_model_name: str = "ExactModel"
    experiment_name: str = "Experiment"
    
    # history for plots
    history_mse_loss: list = field(default_factory=list)
    history_spo_loss: list = field(default_factory=list)
    history_regret: list = field(default_factory=list)


class DeBERTaCVRPTrainer:
    # orchestrates the mse warm up and spo plus fine tuning
    def __init__(
        self, 
        model: nn.Module, 
        opt_model_exact, 
        opt_model_relaxed,
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        config: TrainConfig
    ):
        self.model = model.to(config.device)
        self.opt_model_exact = opt_model_exact      
        self.opt_model_relaxed = opt_model_relaxed  
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # absolutely silence gurobi for the evaluation phases to boost speed
        if hasattr(self.opt_model_exact, '_model'):
            self.opt_model_exact._model.Params.OutputFlag = 0
        if self.opt_model_relaxed is not None and hasattr(self.opt_model_relaxed, '_model'):
            self.opt_model_relaxed._model.Params.OutputFlag = 0

        self.optimizer = torch.optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': self.config.lr_encoder},
            {'params': self.model.edge_head.parameters(), 'lr': self.config.lr_head}
        ])

        self.mse_loss = nn.MSELoss()
        # spo plus uses the relaxed model for fast forward and backward passes
        if self.opt_model_relaxed is not None:
            self.spo_loss = pyepo.func.SPOPlus(self.opt_model_relaxed, processes=self.config.num_processes)

    def _prepare_inputs(self, batch: dict):
        # extracts and formats tensor dimensions from the dataloader
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        if input_ids.dim() == 4: 
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
        return input_ids, attention_mask

    def _train_mse_epoch(self, epoch: int) -> float:
        # executes one epoch of mse warm up
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids, attention_mask = self._prepare_inputs(batch)
            costs = batch["costs"].to(self.device)

            self.optimizer.zero_grad()
            c_pred = self.model(input_ids, attention_mask)
            
            loss = self.mse_loss(c_pred, costs)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % self.config.log_interval == 0 or (batch_idx + 1) == num_batches:
                print(f"      [epoch {epoch} | batch {batch_idx + 1}/{num_batches}] mse loss: {loss.item():.4f}")
                
        return total_loss / num_batches

    def _train_spo_epoch(self, epoch: int) -> float:
        # executes one epoch of decision focused learning
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids, attention_mask = self._prepare_inputs(batch)
            costs = batch["costs"].to(self.device)
            w_star = batch["w_star"].to(self.device)
            z_star = batch["z_star"].to(self.device)

            self.optimizer.zero_grad()
            c_pred = self.model(input_ids, attention_mask)
            
            # gradient calculation via the relaxed model
            loss = self.spo_loss(c_pred, costs, w_star, z_star)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            if (batch_idx + 1) % self.config.log_interval == 0 or (batch_idx + 1) == num_batches:
                print(f"      [epoch {epoch} | batch {batch_idx + 1}/{num_batches}] spo+ loss: {loss.item():.4f}")
                
        return total_loss / num_batches

    def _evaluate(self, epoch: int) -> float:
        # evaluates the model by calculating the decision regret
        self.model.eval()
        total_regret = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                input_ids, attention_mask = self._prepare_inputs(batch)
                costs = batch["costs"].cpu().numpy()
                z_star = batch["z_star"].cpu().numpy()

                c_pred = self.model(input_ids, attention_mask).cpu().numpy()

                for i in range(len(c_pred)):
                    # evaluation must be done with the exact solver
                    self.opt_model_exact.setObj(c_pred[i])
                    w_pred, _ = self.opt_model_exact.solve()
                    
                    actual_cost = np.dot(costs[i], w_pred)
                    total_regret += (actual_cost - z_star[i])
                    
                if (batch_idx + 1) % self.config.log_interval == 0 or (batch_idx + 1) == num_batches:
                    print(f"      [epoch {epoch} eval | batch {batch_idx + 1}/{num_batches}] solving exact regret...")

        return total_regret / len(self.val_loader.dataset)

    def _plot_and_save_metrics(self):
        # generate performance graph and export raw data for future multi experiment benchmarks
        plot_dir = PROJECT_ROOT / "plots" / "textual_dfl" / self.config.nlp_model_name / self.config.experiment_name
        os.makedirs(plot_dir, exist_ok=True)
        
        epochs_mse = range(1, self.config.epochs_mse + 1)
        epochs_spo = range(self.config.epochs_mse + 1, self.config.epochs_mse + self.config.epochs_spo + 1)
        
        # build and save the comparative dataframe
        metrics_df = pd.DataFrame({
            "epoch": range(1, self.config.epochs_mse + self.config.epochs_spo + 1),
            "exact_regret": self.config.history_regret,
            "mse_loss": self.config.history_mse_loss + [np.nan] * self.config.epochs_spo,
            "spo_loss": [np.nan] * self.config.epochs_mse + self.config.history_spo_loss
        })
        metrics_df.to_csv(plot_dir / "metrics.csv", index=False)
        
        # plot and save the png for this specific training run
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss (MSE & SPO+)", color="tab:blue")
        ax1.plot(epochs_mse, self.config.history_mse_loss, 'b--', label="MSE Loss (Warm-up)")
        
        if self.config.epochs_spo > 0:
            ax1.plot(epochs_spo, self.config.history_spo_loss, 'b-', label="SPO+ Loss")
            
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.legend(loc="upper left")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Decision Regret (Exact)", color="tab:red")
        ax2.plot(range(1, len(self.config.history_regret) + 1), self.config.history_regret, 'r-', linewidth=2, label="Regret")
        ax2.tick_params(axis='y', labelcolor="tab:red")
        ax2.legend(loc="upper right")

        plt.title(f"{self.config.nlp_model_name} | Eval: {self.config.exact_model_name} | Exp: {self.config.experiment_name}")
        fig.tight_layout()
        
        plot_path = plot_dir / "training_metrics.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Metrics saved to {plot_dir}/")

    def fit(self):
        save_dir = PROJECT_ROOT / "models" / "textual_dfl" / self.config.nlp_model_name / self.config.experiment_name
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_regret = float('inf')

        def save_checkpoint(epoch, phase):
            print(f"      [Checkpoint] New best regret found! Saving model weights to disk...")
            self.model.encoder.save_pretrained(os.path.join(save_dir, "lora_encoder"))
            torch.save(self.model.edge_head.state_dict(), os.path.join(save_dir, "edge_head.pth"))
            self._plot_and_save_metrics()

        # stage 1: MSE Warm-up
        print(f"--- Stage 1: MSE Warm-up ({self.config.epochs_mse} epochs) ---")
        for epoch in range(1, self.config.epochs_mse + 1):
            start_time = time.time()
            train_loss = self._train_mse_epoch(epoch)
            regret = self._evaluate(epoch)
            
            self.config.history_mse_loss.append(train_loss)
            self.config.history_regret.append(regret)
            epoch_time = time.time() - start_time
            print(f">>> Epoch {epoch} complete | Time: {epoch_time:.1f}s | MSE: {train_loss:.4f} | Exact Regret: {regret:.4f}\n")
            
            # Checkpoint MSE
            if regret < best_val_regret:
                best_val_regret = regret
                save_checkpoint(epoch, "MSE")

        # stage 2: SPO+ Fine-tuning
        if self.config.epochs_spo > 0:
            print(f"--- Stage 2: SPO+ Fine-tuning ({self.config.epochs_spo} epochs) ---")
            for epoch in range(1, self.config.epochs_spo + 1):
                start_time = time.time()
                actual_epoch = self.config.epochs_mse + epoch
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                train_loss = self._train_spo_epoch(actual_epoch)
                regret = self._evaluate(actual_epoch)
                
                self.config.history_spo_loss.append(train_loss)
                self.config.history_regret.append(regret)
                epoch_time = time.time() - start_time
                print(f">>> Epoch {actual_epoch} complete | Time: {epoch_time:.1f}s | SPO+: {train_loss:.4f} | Exact Regret: {regret:.4f}\n")
                
                # checkpoint SPO+
                if regret < best_val_regret:
                    best_val_regret = regret
                    save_checkpoint(actual_epoch, "SPO+")

        print("\nTraining workflow finalized. Check the models folder for the best weights.")

if __name__ == "__main__":
    num_clients = 10
    num_vehicules = 3
    num_nodes = num_clients + 1
    
    rnd = np.random.RandomState(135)
    demands = np.zeros(num_nodes, dtype=int)
    demands[1:] = rnd.randint(1, 10, size=num_clients)
    capacity = math.ceil((np.sum(demands) / num_vehicules) * 1.2)
    
    # initialization of components
    opt_model_exact = cvrpMTZModel(
        num_nodes=num_nodes, 
        demands=demands, 
        capacity=capacity, 
        num_vehicles=num_vehicules
    )
    
    opt_model_relaxed = cvrpMTZModelRel(
        num_nodes=num_nodes, 
        demands=demands, 
        capacity=capacity, 
        num_vehicles=num_vehicules
    )
    
    model = DeBERTaCVRP1(num_nodes=num_nodes)

    # define the experiment name dynamically
    experiment_name = f"SPO_{opt_model_relaxed.__class__.__name__}" 

    config = TrainConfig(
        nlp_model_name=model.__class__.__name__, 
        exact_model_name=opt_model_exact.__class__.__name__, 
        experiment_name=experiment_name
    )

    full_csv_path = PROJECT_ROOT / "data" / "synthetic" / "textual" / "dataset_cvrp_nlp.csv"
    train_csv_path = PROJECT_ROOT / "data" / "synthetic" / "textual" / "train_temp.csv"
    val_csv_path = PROJECT_ROOT / "data" / "synthetic" / "textual" / "val_temp.csv"

    if not os.path.exists(train_csv_path):
        df = pd.read_csv(full_csv_path)
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

    # instantiate datasets which will safely hit the cache or generate it once
    train_dataset = cvrpTextDataset(csv_file=str(train_csv_path), opt_model=opt_model_exact)
    val_dataset = cvrpTextDataset(csv_file=str(val_csv_path), opt_model=opt_model_exact)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # execution
    trainer = DeBERTaCVRPTrainer(
        model=model, 
        opt_model_exact=opt_model_exact, 
        opt_model_relaxed=opt_model_relaxed, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        config=config
    )
    
    trainer.fit()