import paddle
import paddle.nn as nn
from paddle.io import DataLoader,TensorDataset
from utils.model import distillmodel
from collections import namedtuple
# import paddlenlp
# import random
from paddlenlp.transformers import GPTTokenizer
paddle.device.set_device("gpu:6")

# tokenizer=GPTTokenizer.from_pretrained("gpt2-xl")

# a="Sentiment: I love you."
# # b="\nSentiment: I love you."
# # c="I love you."
# # d=" I love you."

# print(tokenizer.encode(" positive"))
# print(tokenizer.encode(b))
# print(tokenizer.encode(c))
# print(tokenizer.encode(d))

class net(nn.Layer):
    def __init__(self,dim):
        super(net,self).__init__()
        self.w=nn.Linear(dim,dim)
        self.w2=nn.Linear(dim*2,dim*2)
    def forward(self,x,y):
        y1=self.w(x)
        x2=paddle.concat((y,y1),axis=1)
        return self.w2(x2)
    
a=net(5)

x1=paddle.rand((2,5))
x2=paddle.rand((2,5))
y1=a(x1,x2)

print(y1)