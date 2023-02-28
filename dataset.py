import torch
import numpy as np
import pickle
import math
class CharDS(torch.utils.data.Dataset):

    def __init__(self, txt_pth:str = 'data/train.txt', test_pth:str = 'data/test.txt', seq_size: int = 512, train: bool=True):
        self.txt=''
        pth=''
        if train:
            pth=txt_pth
        else:
            pth=test_pth
        with open(pth, encoding='utf-8-sig') as f:
            self.txt = f.read()
        self.chars = sorted(list(set(self.txt)))
        self.ch2idx = {c:i for (i, c) in enumerate(self.chars)}        
        self.idx2ch = {i:c for (i, c) in enumerate(self.chars)}        
        self.num_chars=len(self.chars)
        self.seq_size=seq_size        
        self.datasize = len(self.txt)

        if train:
            with open('checkpoints/embedding.pickle', 'wb') as f:
                pickle.dump({'ch2idx': self.ch2idx, 'idx2ch':self.idx2ch, 'size':self.num_chars}, f)

    def __len__(self):
        return math.ceil(self.datasize / self.seq_size )

    def __getitem__(self, idx):     
        idx = (idx + len(self)) % len(self)
        beg = idx*self.seq_size
        x = np.zeros((self.seq_size, self.num_chars))
        y = np.zeros(self.seq_size)
        for n in range(self.seq_size+1):            
            i = beg + n
            c=''
            if i < self.datasize:
                c=self.txt[i]
            else:
                c='\0'
            if n < self.seq_size:
                x[n][self.ch2idx[c]] = 1
            if n > 0:
                y[n-1] = self.ch2idx[c]


        return (torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long))

class SeqSampler(torch.utils.data.Sampler):

    def __init__(self, ds, bs=64):
        self.bs = bs
        self.ds = ds
        self.idx=999999999999999999999
    
    def __iter__(self):
        self.idx=0
        return self

    def __len__(self):
        return len(self.ds)

    def __next__(self):
        if self.idx >= len(self.ds):
            raise StopIteration
        pt = int(((self.idx % self.bs)/self.bs) * len(self.ds)) + (self.idx // self.bs)
        self.idx += 1
        return pt

def str_to_tensor(inp, ch2idx, one_hot=True):
    ret=None
    if one_hot:
        ret = np.zeros((len(inp), len(ch2idx)))
        for (i, c) in enumerate(inp):
            ret[i][ch2idx[c]]=1
    else:
        ret = np.zeros(len(inp))
        for i in range(len(inp)):
            ret[i]=ch2idx[inp[i]]
    return torch.Tensor(ret)
def tensor_to_str(tensor, idx2ch):
    l=[]
    if len(tensor.shape) > 1:
        for i in range(tensor.shape[0]):
            idx = torch.argmax(tensor[i]).item()
            c=idx2ch[idx]
            if c== '\0':
                break
            l.append(c)
    else:
        for idx in tensor:
            c=idx2ch[idx.item()]
            if c== '\0':
                break
            l.append(c)
    return "".join(l)




def main():
    ds = CharDS()
    smplr = SeqSampler(ds)
    print(len(ds))
    print(len(smplr))
    print(tensor_to_str(str_to_tensor("caneta azuuuul, azuuul caneta", ds.ch2idx), ds.idx2ch))
    print(tensor_to_str(str_to_tensor("caneta azuuuul, azuuul caneta", ds.ch2idx, one_hot=False), ds.idx2ch))
    for s in smplr:
        print(s)
        print('.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
        print(tensor_to_str(ds[s][0], ds.idx2ch))
        print('.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
        print(tensor_to_str(ds[s][1], ds.idx2ch))
        input()
   

if __name__ == "__main__":
    main()