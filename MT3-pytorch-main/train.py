import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config.data_config import data_config
from config.T5config import Magenta_T5Config
from model.T5 import Transformer
from data.MAESTRO_loader import MIDIDataset, MIDIDatasetTab
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.constants import *
import gc

import os
from pytorch_lightning.callbacks import ModelCheckpoint
from metrics import f1_from_tokens_rough

device = "cuda" if torch.cuda.is_available() else "cpu"

experiment_config={
        "learning_rate": 1e-4, #1e-3 in article
        "architecture": "Magenta",
        "training_steps": 1000000, #originaly 1000000
        "checking_steps": 100000,
        "batch": 16, #256 in article, 16 originally in code
        "kernel_size": 0,
        "expansion_factor": 0
        }

class MT3Trainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Transformer(config=Magenta_T5Config)
        #self.criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
        self.criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_TAB)
        self.cpt_path = data_config.cpt_path
        self.val_outputs = []
        os.makedirs(self.cpt_path, exist_ok=True)

    def forward(self, encoder_input_tokens, decoder_target_tokens, decode):
        return self.model.forward(encoder_input_tokens, decoder_target_tokens, decoder_input_tokens=None)
    
    def training_step(self, batch, batch_idx):
        #print(f"Training step {batch_idx}, batch size: {batch['inputs'].shape[0]}")
        inputs = batch['inputs']
        targets = batch['targets']
        #print(f"Targets min: {targets.min()}, max: {targets.max()}")  # Debugging
        outputs = self.forward(encoder_input_tokens=inputs, decoder_target_tokens=targets, decode=False)
        loss = self.criterion(outputs.permute(0,2,1), targets)
        #print(batch_idx) looks like 60 trainign steps every epoch, resets mid epochs
        #use self.current_epoch for correct
        if (batch_idx + 1) % (experiment_config['checking_steps'] / experiment_config['batch']) == 0:
            print("----------Saving checkpoint------------")
            torch.save(self.model.state_dict(), self.cpt_path + experiment_config['architecture'] + '/' + str((batch_idx+1)*experiment_config['batch'])+'.ckpt')
            #added self. to model
        self.log("train/loss", loss)
        
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        #print("--------validation---------")
        inputs = batch['inputs']
        targets = batch['targets']
        #print(f"Targets min: {targets.min()}, max: {targets.max()}")  # Debugging
        outputs = self.forward(encoder_input_tokens=inputs, decoder_target_tokens=targets, decode=False)
        loss = self.criterion(outputs.permute(0,2,1), targets)

        self.log("val/loss", loss)

        # Get predicted tokens (e.g., argmax over vocab dimension)
        preds = outputs.argmax(dim=-1).cpu().tolist()
        targets = targets.cpu().tolist()

        #added cause newer pytorch lightning
        self.val_outputs.append({"preds": preds, "targets": targets})

        return {"loss": loss.cpu().item(), "preds": preds, "targets": targets}
        #return loss

    def on_validation_epoch_end(self):
        #added fornew lighnign pytorch
        outputs = self.val_outputs
        # Aggregate all predictions and targets
        all_preds = []
        all_targets = []
        for out in outputs:
            all_preds.extend(out["preds"])
            all_targets.extend(out["targets"])
        # Compute F1 for each pair and average
        f1s = []
        for gt, pred in zip(all_targets, all_preds):
            _, _, f1 = f1_from_tokens_rough(gt, pred)
            f1s.append(f1)
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        # Log to wandb
        self.log("val/f1", avg_f1, prog_bar=True, logger=True)
        
        #added cause newer pytorch liughing
        self.val_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), experiment_config['learning_rate'])
        
        return optimizer
    
    def train_dataloader(self):
        #train_data = MIDIDataset(type='train')
        train_data = MIDIDatasetTab()
        trainloader = DataLoader(train_data, batch_size=experiment_config["batch"], num_workers=4)
        return trainloader
    
    def val_dataloader(self):
        #validation_data = MIDIDataset(type='validation')
        validation_data = MIDIDatasetTab(type='validation')
        validloader = DataLoader(validation_data, batch_size=experiment_config["batch"], num_workers=4)
        return validloader
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()

    checkpoint_callback = ModelCheckpoint(
    dirpath='mt3-pytorch',
    filename='epoch={epoch:04d}',
    save_top_k=-1,  # Save all checkpoints
    every_n_epochs=100,
    save_weights_only=True
    )
    
    model = MT3Trainer()

    """train_data = MIDIDatasetTab()
    print(f"Dataset length: {len(train_data)}")  # Should be > batch size, else epoch is too small

    trainloader = DataLoader(train_data, batch_size=experiment_config['batch'], num_workers=4)
    print(f"Number of batches in trainloader: {len(trainloader)}")

    for i, batch in enumerate(trainloader):
        print(f"Batch {i}: inputs shape {batch['inputs'].shape}")
        if i > 10:
            break"""
    
    print(model)
    print(experiment_config)

    wandb_logger = WandbLogger(project="mt3-pytorch")

    import wandb

    trainer = pl.Trainer(accelerator='gpu',    # ✅ Use accelerator instead of `gpus`
                         devices=1,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         max_steps=experiment_config['training_steps'],
                         callbacks=[checkpoint_callback],
                         max_epochs=10000
                         #limit_train_batches=60,  # or whatever number of batches per epoch you want
                         )
                         #changed check val each epoch to 1 from experiment_config["checking_steps"]
                         #changed gpus=1 to devices=2, added accečeratpr="gpu" and strategy ="ddp"
    trainer.fit(model)

    wandb.save("train.py")
    wandb.save("config/T5config.py")
    wandb.save("data/constants.py")
    wandb.save("data/MAESTRO_loader.py")

    #neki ni uredu z epochi, skos pise epoch 0 -> zrihtu, len funkcija u dataloaderju