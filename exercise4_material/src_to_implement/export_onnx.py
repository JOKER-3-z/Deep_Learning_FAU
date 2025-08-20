import torch as t
from trainer import Trainer
import sys
import torchvision as tv
from model import ResNet

def export(epoch):
    epoch = int(epoch)
    model=ResNet()
    crit = t.nn.BCELoss()
    trainer = Trainer(model, crit)
    trainer.restore_checkpoint(epoch)
    trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))

