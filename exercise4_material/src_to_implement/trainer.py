import torch as t
import os
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

class Trainer:
    def __init__(self, model, crit, optim=None, train_dl=None, val_test_dl=None, cuda=True, early_stopping_patience=-1):
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        os.makedirs("checkpoints", exist_ok=True)
        t.save({'state_dict': self._model.state_dict()}, f'checkpoints/checkpoint_{epoch:03d}.ckp')

    def restore_checkpoint(self, epoch_n):
        ckp = t.load(f'checkpoints/checkpoint_{epoch_n:03d}.ckp', 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m, x, fn,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        self._optim.zero_grad()
        output = self._model(x)
        loss = self._crit(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def val_test_step(self, x, y):
        y_p = self._model(x)
        loss = self._crit(y_p, y.float())
        return loss.item(), y_p

    def train_epoch(self):
        self._model.train()
        total_loss = 0
        for x, y in tqdm(self._train_dl, desc="Training", leave=False, dynamic_ncols=True):
            if self._cuda:
                x, y = x.cuda(), y.cuda()
            total_loss += self.train_step(x, y)
        return total_loss / len(self._train_dl)

    def val_test(self):
        self._model.eval()
        y_pre = []
        y_true = []
        total_loss = 0
        with t.no_grad():
            for x, y in tqdm(self._val_test_dl, desc="Validating", leave=False, dynamic_ncols=True):
                if self._cuda:
                    x, y = x.cuda(), y.cuda()
                loss, y_p = self.val_test_step(x, y)
                total_loss += loss
                y_pre.append(y_p)
                y_true.append(y)
        y_pre = t.cat(y_pre, dim=0)
        y_true = t.cat(y_true, dim=0)
        y_pred_label = (y_pre > 0.5).int()
        avg_loss = total_loss / len(self._val_test_dl)
        exact_match_acc = (y_pred_label == y_true.int()).all(dim=1).float().mean().item()
        y_true, y_pred_label = y_true.cpu().numpy(), y_pred_label.cpu().numpy()
        f1_micro = f1_score(y_true, y_pred_label, average='micro', zero_division=0)
        print(f"[Validation] Loss: {avg_loss:.4f} | ExactMatch Acc: {exact_match_acc:.4f} | F1_micro: {f1_micro:.4f}")
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses = []
        eval_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        epoch = 0

        while True:
            epoch += 1
            print(f"\nEpoch {epoch}:")
            if epochs > 0 and epoch >= epochs:
                break
            t_loss = self.train_epoch()
            v_loss = self.val_test()
            train_losses.append(t_loss)
            eval_losses.append(v_loss)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                self.save_checkpoint(epoch)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if self._early_stopping_patience > 0 and epochs_no_improve >= self._early_stopping_patience:
                print("Early stopping triggered.")
                break
        return train_losses, eval_losses
