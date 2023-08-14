import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch import Tensor

import torch
import zero

import modules


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def apply_model(device, model, x_num, x_cat=None):
    if isinstance(model, modules.FTTransformer):
        x_num = x_num.to(torch.float)
        return model(x_num, x_cat).to(device)
    elif isinstance(model, (modules.MLP, modules.ResNet)):
        assert x_cat is None

        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(mode, model, optimizer, X, y, part, checkpoint_path, device):
    
    if mode == 'validation':
      if checkpoint_path:
          checkpoint = torch.load(checkpoint_path, map_location=device)

          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          epoch = checkpoint['epoch']
          loss = checkpoint['loss']

    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 256):
        prediction.append(apply_model(device, model, batch))

    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    #print(target.shape, prediction.shape)

    mse = mean_squared_error(target, prediction)
    #valid_loss = criterion(target, prediction)
    score = np.sqrt(mse)

    return score, mse

class SaveBestModel:

    def __init__(
        self, checkpoint_path=str(''), best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.checkpoint_path = checkpoint_path

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, self.checkpoint_path)
            
def save_model(epochs, model, optimizer, criterion, model_path):
    
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, model_path)

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.savefig('outputs/accuracy.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('outputs/loss.png')