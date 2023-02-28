from dataset import CharDS, tensor_to_str, SeqSampler
from model import LSTMNet
from torch.utils.data import DataLoader
import torch
import os

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    b=0
    bs=16
    ss=100
    layers=2
    hidden_size=512
    dropout=0.3    
    epochs=6000
    check_each=600
    epoch = 0

    embedding = {}    
    ds=CharDS(seq_size=ss)    
    loss = torch.nn.CrossEntropyLoss()
    model = LSTMNet(ds.num_chars, lstm_layers=layers, hidden_size=hidden_size, dropout=dropout).to(device)

    crit = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss_hist=[]
    test_hist=[]
    
    if os.path.exists('checkpoints/train_hist.pth'):
        hist = torch.load('checkpoints/train_hist.pth')
        loss_hist=hist['loss_hist']
        epoch = hist['num_epochs']
        num_batches= hist['num_batches']
        npe = hist['batches_per_epoch']
        
        b= num_batches % npe + 1
        if os.path.exists(f'checkpoints/{num_batches}.pth'):
            chk = torch.load(f'checkpoints/{num_batches}.pth')
            model.load_state_dict(chk['model'])
            optimizer.load_state_dict(chk['optim'])    
        
    dl = DataLoader(ds, batch_size=bs, sampler=SeqSampler(ds, bs), drop_last=True)
    n = len(dl)
    test_ds = CharDS(seq_size=ss, train=False)
    test_dl = DataLoader(ds, batch_size=bs, sampler=SeqSampler(test_ds, bs), drop_last=True)

    def checkpoint(model, epoch, crit):
        loss=0.0
        model.eval()
        h, c = model.init_state(bs)    
        with torch.no_grad():
            for _, (x, y) in enumerate(test_dl):
                x=x.to(device)
                y=y.to(device)

            
                y_pred, (h, c) = model(x, (h, c))

                loss += crit(y_pred.transpose(1, 2), y).item() / len(test_dl)
          
        print(f"TEST LOSS: {loss}")
        test_hist.append(loss)
        torch.save({'model':model.state_dict(), 'emb_size':ds.num_chars, 'lstm_layers':layers, 'hidden_size':hidden_size, 'dropout': dropout, 'optim':optimizer.state_dict()}, f'checkpoints/{len(loss_hist)}.pth')    
        torch.save({'num_batches': len(loss_hist),'batches_per_epoch':n, 'num_epochs': epoch, 'loss_hist':loss_hist, 'test_hist':test_hist}, f'checkpoints/train_hist.pth')
        model.train()

    total_b = len(loss_hist)
    while epoch < epochs:
        h, c = model.init_state(bs)    
        model.train()
        for batch, (x, y) in enumerate(dl):
            if batch < b:
                continue
                      
            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()
            y_pred, (h, c) = model(x, (h, c))

            loss = crit(y_pred.transpose(1, 2), y)

            h=h.detach()
            c=c.detach()

            loss.backward()
            optimizer.step()

            li=loss.item()

            loss_hist.append(li)

            print(f"EPOCH: {epoch} BATCH: {batch}/{n} LOSS: {li}")

            if total_b % check_each == 0:
                checkpoint(model, epoch, crit)

            b += 1
            total_b += 1
        print(f"EPOCH COMPLETE")
        
        epoch += 1
        b = 0
if __name__ == "__main__":
    main()