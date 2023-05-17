import paddle
import paddle.nn as nn
# from paddle.io import DataLoader,TensorDataset
# from utils.model import distillmodel
# from collections import namedtuple
# import random
# from paddlenlp.transformers import GPTTokenizer,GPTLMHeadModel
paddle.device.set_device("gpu:4")

a=paddle.rand([3,5])
b=paddle.rand([3,5])

c=paddle.einsum("ik,jk->ij",a,b)

print(c)

d=paddle.linalg.norm(a,p=2,axis=1)

print(d)

x=c/d.unsqueeze(1)
y=c/d.unsqueeze(0)

print(x)
print(y)
# class distillmodel(nn.Layer):
#     def __init__(self, llm_path):
#         super(distillmodel,self).__init__()
#         """
#         init LLM and single MLP layer
#         """
#         self.LLM=GPTLMHeadModel.from_pretrained(llm_path)
#         #k special token
#         vocab_size=self.LLM.config.vocab_size
#         self.special_token=paddle.create_parameter([4,self.LLM.config.hidden_size],dtype="float32")
#         #sample 4 random token in LLM embedding to initialize
#         ids=random.sample([i for i in range(vocab_size)],4)
#         ids=paddle.to_tensor(ids)
#         self.tensor=self.get_embedding(ids).detach()
#         nn.utils.vector_to_parameters(self.tensor.reshape((1,-1)).squeeze(0),[self.special_token])
    
#     def get_embedding(self, input_ids):
#         """
#         get LLM embedding for each input ids
#         """
#         return self.LLM.get_input_embeddings()(input_ids)

# a=distillmodel("gpt2-xl")

# print()




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
