from models.builder import build_network
from spikerplus import Trainer
from data.dataloader import get_dataloaders
from utils.config import cfg
import os

snn = build_network(cfg)
train_loader, val_loader = get_dataloaders(cfg)
trainer = Trainer(snn, cfg=cfg,predict_filter=cfg.predict_filter)
print(f"Training with model type: {cfg.model_type}")
checkpoint_path = "checkpoints/spiking-fsb-conv-light/checkpoint_e80.pth"
checkpoint_flag = True # Set to True if you want to resume from a checkpoint
if os.path.exists(checkpoint_path) and checkpoint_flag:
    print("Checkpoint found, continuing the training...")
    start_epoch = trainer.resume_from_checkpoint(checkpoint_path)
else:
    start_epoch = 1


# for param_group in trainer.optimizer.param_groups:
#     param_group['lr'] = 1e-4
# print(f"Manually set learning rate to {trainer.optimizer.param_groups[0]['lr']}")

trainer.train(
    train_loader,
    val_loader,
    n_epochs=cfg.n_epochs,
    store=True,
    encode_mode=cfg.encode_mode,
    max_len=cfg.max_len,
    start_epoch=start_epoch
)
