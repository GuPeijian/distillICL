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

# class net(nn.Layer):
#     def __init__(self,dim):
#         super(net,self).__init__()
#         self.w=nn.Linear(dim,dim)
#         self.w2=nn.Linear(dim*2,dim*2)
#     def forward(self,x,y):
#         y1=self.w(x)
#         x2=paddle.concat((y,y1),axis=1)
#         return self.w2(x2)
    
# a=net(5)

x1=paddle.rand((2,5))

x2=paddle.rand((2,5))

p1=nn.functional.softmax(x1,axis=1)

p2=nn.functional.softmax(x2,axis=1)

log_p1=paddle.log(p1)
log_p2=paddle.log(p2)

kl_d=nn.functional.kl_div(log_p1,p2)

print(kl_d)
l=paddle.mean((log_p2-log_p1)*p2)
k=paddle.mean(paddle.sum((log_p2-log_p1)*p2,axis=1))

print(l)
