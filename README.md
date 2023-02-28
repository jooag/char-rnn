# Char RNN

LSTM implemented in PyTorch capable of learning structure of texts char by char.

Use:
Data should be in train.txt and test.txt. Hyperparameters can be set at train.py source code.
 ```
 python train.py
 ```
Ctrl + C to stop training.

To generate text:
```
python generate_text.py $LEN
```
Where $LEN is the number of characters to be generated. Enter the start of the text ended with an '#'. Press Enter to generate more.
