import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # load the data from the csv file and perform a train-test-split
    csv_file = "./data.csv"
    df = pd.read_csv(csv_file,sep=';')
    print("data loading……")
    # this can be accomplished using the already imported pandas and sklearn.model_selection modules
    train_df,val_df = train_test_split(df,test_size=0.2,random_state=42, stratify=df[['crack', 'inactive']])
    print("len of train dataset: "+str(len(train_df))+" len of eval dataset: "+str(len(val_df)))
    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    train_dl = t.utils.data.DataLoader(ChallengeDataset(train_df, 'train'), batch_size=512)
    val_dl = t.utils.data.DataLoader(ChallengeDataset(val_df, 'val'), batch_size=512)
    print("model define……")
    # create an instance of our ResNet model
    resnet = model.ResNet()

    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    crit = t.nn.BCELoss()
    # set up the optimizer (see t.optim)
    optim = t.optim.Adam(resnet.parameters(),lr=1e-4,weight_decay=1e-5)
    # create an object of type Trainer and set its early stopping criterion
    trainer = Trainer(
                model = resnet,                       #model
                crit = crit,                         # Loss function
                optim = optim,                   # Optimizer
                train_dl = train_dl,                # Training data set
                val_test_dl = val_dl,             # Validation (or test) data set
                cuda=True,                    # Whether to use the GPU
                early_stopping_patience=5
            )

    # go, go, go... call fit on trainer
    print("training……")
    res = trainer.fit(20)

    # plot the results
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('./losses.png')