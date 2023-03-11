import os
import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

# PyTorch Lightning Modules
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='./../')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=1)

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    exp_name = '%s-%s'%(hp.backbone_name, hp.dataset_name)
    logger = WandbLogger(project="Baseline FGSBIR", name=exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='top1', mode='max', dirpath=exp_name,
        filename=hp.backbone_name, save_last=True)
    
    if os.path.exists(os.path.join(exp_name, hp.backbone_name, 'last.ckpt')):
        ckpt_path = os.path.join(exp_name, hp.backbone_name, 'last.ckpt')
        model = FGSBIR_Model(hp).load_from_checkpoint(
            checkpoint_path=ckpt_path)
    else:
        model = FGSBIR_Model(hp)
        ckpt_path = None

    profiler = SimpleProfiler(
        dirpath=os.path.join(exp_name, hp.dataset_name),
        filename='perf-logs')
    
    trainer = Trainer(logger=logger,
        accelerator='gpu', devices=1, accumulate_grad_batches=None,
        benchmark=False, deterministic=False, detect_anomaly=False,
        callbacks=[checkpoint_callback], check_val_every_n_epoch=1,
        log_every_n_steps=10, overfit_batches=0.0, limit_val_batches=1.0,
        max_epochs=hp.max_epoch, enable_model_summary=True, profiler=profiler)

    trainer.fit(model, dataloader_Train, dataloader_Test, ckpt_path=ckpt_path)
