import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
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
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        output = self._model(x)
        # -calculate the loss
        loss = self._crit(output,y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        # predict
        y_p = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(y_p, y.float())
        # return the loss and the predictions
        return loss.item(), y_p
        
        
    def train_epoch(self):
        # set training mode
        self._model.train()
        total_loss = 0
        # iterate through the training set
        for x,y in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x,y = x.cuda(),y.cuda()
            # perform a training step
            total_loss += self.train_step(x,y)
        # calculate the average loss for the epoch and return it
        return total_loss/len(self._train_dl)


    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        y_pre = []
        y_true = []
        total_loss = 0
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        with t.no_grad():
             # iterate through the validation set
            for x,y in self._val_test_dl:
                # transfer the batch to the gpu if given
                if self._cuda:
                    x,y = x.cuda(),y.cuda()
                    # perform a validation step
                    loss,y_p = self.val_test_step(x,y)
                    # save the predictions and the labels for each batch
                    total_loss+=loss
                    y_pre.append(y_p)
                    y_true.append(y)
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        y_pre = t.cat(y_pre, dim=0)
        y_true = t.cat(y_true, dim=0)
        # 二值化预测（0.5为阈值）
        y_pred_label = (y_pre > 0.5).int()

        # 计算平均 loss
        avg_loss = total_loss / len(self._val_test_dl)
        # 多标签准确率（按实例完全匹配）
        exact_match_acc = (y_pred_label == y_true.int()).all(dim=1).float().mean().item()

        # Micro 和 Macro F1
        f1_micro = f1_score(y_true, y_pred_label, average='micro', zero_division=0)#对常见类别友好
        f1_macro = f1_score(y_true, y_pred_label, average='macro', zero_division=0)#更关注少量样本
        print(f"[Validation] Loss: {avg_loss:.4f} | ExactMatch Acc: {exact_match_acc:.4f} | F1_micro: {f1_micro:.4f} | F1_macro: {f1_macro:.4f}")
        return avg_loss
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        eval_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        epoch = 0
        
        while True:
            # stop by epoch number
            epoch += 1
            if epochs > 0 and epoch >= epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            t_loss = self.train_epoch()
            v_loss = self.val_test()
            # append the losses to the respective lists
            train_losses.append(t_loss)
            eval_losses.append(v_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                self.save_checkpoint(epoch)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if self._early_stopping_patience > 0 and epochs_no_improve >= self._early_stopping_patience:
                break
        # return the losses for both training and validation
        return train_losses, eval_losses
                    
        
        
        
