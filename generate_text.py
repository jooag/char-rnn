from model import LSTMNet
from dataset import tensor_to_str, str_to_tensor
import sys
import pickle
import torch
import numpy as np

def main():
    temp=0.1
    start = []
    num=64
    if len(sys.argv) > 1:
        num=int(sys.argv[1])
    c = sys.stdin.read(1)
    while c != '#':
        start.append(c)
        c = sys.stdin.read(1)
    start="".join(start)
    emb = None
    with open('checkpoints/embedding.pickle', 'rb') as f:
        emb = pickle.load(f)

    
    train_data=torch.load('checkpoints/train_hist.pth')
    n = train_data['num_batches']
    
    chk = torch.load(f'checkpoints/{n}.pth')
    model = LSTMNet(emb_size=chk['emb_size'], hidden_size=chk['hidden_size'], lstm_layers=chk['lstm_layers'], dropout=chk['dropout'])
    
    
    model.load_state_dict(chk['model'])   

    start_tensor=str_to_tensor(start, emb['ch2idx'])
    idx2ch = emb['idx2ch']

    (h, c) = model.init_state()

    out, (h, c) = model(start_tensor, (h, c))
    y = torch.zeros((1, len(idx2ch)))
    y[0] = out[-1]
    print('\b\b', end='')
    model.eval()
    print(start, end='')
    with torch.no_grad():
        while True:
            for i in range(num):          
                prob=torch.nn.functional.softmax(y[0]/temp, dim = 0).numpy()                
                i = np.random.choice(y.shape[1], None, p=prob)
                print(idx2ch[i], end='')
                x=torch.zeros_like(y)
                x[0,i]=1
                y, (h, c) = model(x, (h, c))
                sys.stdout.flush()
            input()
       
if __name__ == "__main__":
    main()