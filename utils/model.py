import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddlenlp.transformers import GPTLMHeadModel

class distillmodel(nn.Layer):
    def __init__(self, config):
        super(distillmodel,self).__init__()
        """
        init LLM and single MLP layer
        """
        self.LLM=GPTLMHeadModel.from_pretrained(config.llm_path)
        self.hidden_size=self.LLM.config.hidden_size
        self.mlp_hidden_size=self.hidden_size//4
        #TODO add length truncation in source and process label positions
        #self.sample_length=config.sample_length
        self.MLP=nn.Sequential(
            nn.Linear(self.hidden_size,self.mlp_hidden_size),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_size,self.hidden_size)
        )
        self.config=config
        

    def get_embedding(self, input_ids):
        """
        get LLM embedding for each input ids
        """
        return self.LLM.get_input_embeddings()(input_ids)
    
    def length_reduce(self,inputs_tokens,length,mode="mean"):
        """
        reduce length of inputs_embeds by pooling
        """
        inputs_tokens=paddle.concat(inputs_tokens,axis=1) # [1,N*L*k]
        inputs_embeds=self.get_embedding(inputs_tokens).squeeze(0) # [N*L*k,d]
        inputs_embeds=self.MLP(inputs_embeds)
        start_pos=0
        output=[]
        for _length in length:
            target_embeds=inputs_embeds[start_pos:start_pos+_length,:]
            if mode == "mean":
                outputs_embeds=paddle.mean(target_embeds,axis=0,keepdim=True)
            elif mode == "max":
                outputs_embeds=paddle.max(target_embeds,axis=0,keepdim=True)
            elif mode =="meanmax":
                outputs_embeds=paddle.concat((paddle.mean(target_embeds,axis=0,keepdim=True),
                                            paddle.max(target_embeds,axis=0,keepdim=True)),
                                            axis=1)
            else:
                raise NotImplementedError("pooling method is not implemented")
            output.append(outputs_embeds.unsqueeze(0)) #[1,1/2,d]
            start_pos += _length
        return output

    def icl_predict(self, input_ids, inputs_embeds, attention_mask, label_ids):
        """
        icl predict of LLM
        """
        #need a None in two inputs
        if input_ids is not None and inputs_embeds is not None:
            raise IndentationError("need a None input")

        #predict
        logits = self.LLM.forward(input_ids=input_ids,
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                return_dict=True).logits
        #gather label logits
        label_logits=paddle.gather(logits,label_ids,axis=2)
        return label_logits
        
    def get_logits(self, logits, label_positions):
        """
        get icl predcit logits for each label positions
        label_positions: list of list
        """
        #get gather nd index
        _label_positions=[]
        _index=[]
        for i,positions in enumerate(label_positions):
            _label_positions.extend(positions)
            _index.extend([i * len(positions)])
        
        #turn to CPU tensor
        _index=paddle.to_tensor(_index,place=paddle.CPUPlace())
        _label_positions=paddle.to_tensor(_label_positions,place=paddle.CPUPlace())

        _index=paddle.to_tensor(_index,place=logits.place)
        _label_positions=paddle.to_tensor(_label_positions,place=logits.place)
        _index=_index.unsqueeze(1)
        _label_positions=_label_positions.unsqueeze(1)
        index=paddle.concat((_index,_label_positions),axis=1)

        #get logits
        return paddle.gather_nd(logits,index)

    def calcu_loss(self, logits_s, logits_t, label):
        """
        calculae loss
        consider both distill loss for unlabeled example and CE loss for labeled example
        """
        #KL loss
        prob_s=F.softmax(logits_s,)
        prob_t=F.softmax(logits_t,)
        def kl_loss(p1,p2):
            log_p1=paddle.log(p1)
            log_p2=paddle.log(p2)
            kl_d=paddle.mean(paddle.sum((log_p2-log_p1)*p2,axis=1))
            return kl_d

        kl_st=kl_loss(prob_s,prob_t)
        kl_ts=kl_loss(prob_t,prob_s)
        loss=(kl_st+kl_ts)/2

        #CE loss
        if self.config.CEloss:
            loss_fct=nn.CrossEntropyLoss()
            assert label is not None
            loss_CE=loss_fct(logits_t,label)
            loss = loss + loss_CE * self.config.alpha

        return loss
        
    def padding(self,instruction,input_list,label_position_list):
        """
        padding the combined input
        input is list of [1,L] or [1,L,D] 
        """
        #get length
        input_length=[]
        for input_item in input_list:
            input_length.append(input_item.shape[1])
        
        max_length=max(input_length)
        pad_length=[max_length-i for i in input_length]
        #padding
        output_list=[]
        attention_mask_list=[]
        instruction_length=instruction.shape[1]
        for i,input_item in enumerate(input_list):
            #concat instruction
            output_item=paddle.concat([instruction,input_item],axis=1)
            attention_mask_item=[1]*output_item.shape[1]
            #pad
            if pad_length[i] > 0:
                pad_item=[self.config.pad_token_id] * pad_length[i]
                pad_item=paddle.to_tensor(pad_item,place=paddle.CPUPlace())
                pad_item=paddle.to_tensor(pad_item,place=instruction.place)
                pad_item=pad_item.unsqueeze(0)
                if len(input_item.shape)==3:
                    pad_item=self.get_embedding(pad_item)
                output_item=paddle.concat((pad_item,output_item),axis=1) # left pad
                attention_mask_item=[0]*pad_length[i] + attention_mask_item
            #attention mask
            attention_mask_item=paddle.to_tensor(attention_mask_item,place=paddle.CPUPlace())
            attention_mask_item=paddle.to_tensor(attention_mask_item,place=instruction.place)
            attention_mask_item=attention_mask_item.unsqueeze(0)

            output_list.append(output_item)
            attention_mask_list.append(attention_mask_item)

            #label position
            shift_length=instruction_length+pad_length[i]
            label_position_list[i]=[position+shift_length for position in label_position_list[i]]
        
        output=paddle.concat(output_list,axis=0)
        attention_mask=paddle.concat(attention_mask_list,axis=0)

        return output,attention_mask,label_position_list

    def forward(self,instruction_tokens,input_tokens,example_positions,label_ids,label,label_positions,mode="train"):
        """
        complete pipeline
        instruction_tokens: tokenized instruction
        input_tokens: list of list of tokenized text
        example_positions: list of list of index, indexing the position of example index
        label_ids: 1d tensor indexing verbalizer token
        label: label 
        label_positions: list of position for each label piece
        """
        batch_size=len(input_tokens)
        if mode=="train":
            #build common icl input
            combined_input_tokens=[]
            true_positions_s=[]
            for batch_i in range(batch_size):
                _input_tokens=input_tokens[batch_i]
                _example_positions=example_positions[batch_i]
                _label_positions=label_positions[batch_i]
                _combined_input_tokens=[]
                length=[]
                for piece_index, piece in enumerate(_input_tokens):                
                    #simple concat
                    _combined_input_tokens.append(piece)
                    length.append(piece.shape[1])

                _combined_input_tokens=paddle.concat(_combined_input_tokens,axis=1)
                combined_input_tokens.append(_combined_input_tokens)

                #process label
                _true_positions_s=[]
                for label_index in _label_positions:
                    if label_index == -1:
                        #the last token
                        _true_positions_s.append(sum(length)-1) # -1 for causal model
                    else:
                        _true_positions_s.append(sum(length[:label_index])-1) # -1 for causal model

                true_positions_s.append(_true_positions_s)

            #padding
            input_ids_s,attention_mask_s,true_positions_s=self.padding(instruction_tokens,combined_input_tokens,true_positions_s)

        #build reduced icl input
        instruction_embeddings=self.get_embedding(instruction_tokens)
        length=[]
        example_tokens=[]
        for batch_i,_example_positions in enumerate(example_positions):
            for example_i in _example_positions:
                _tokens=input_tokens[batch_i][example_i]
                length.append(_tokens.shape[-1])
                example_tokens.append(_tokens)
        example_embeds=self.length_reduce(example_tokens,length,mode=self.config.reduce_mode)
        example_embeds_iter=iter(example_embeds)
        #combine different part
        reduced_embeds=[]
        true_positions_t=[]
        for batch_i in range(batch_size):
            _input_tokens=input_tokens[batch_i]
            _example_positions=example_positions[batch_i]
            _label_positions=label_positions[batch_i]
            _reduced_embeds=[]
            length=[]
            for piece_index, piece in enumerate(_input_tokens):              
                if piece_index not in _example_positions:
                    #simple concat
                    piece_embeds=self.get_embedding(piece)
                    _reduced_embeds.append(piece_embeds)
                    length.append(piece_embeds.shape[1])
                else:
                    #choose reduced
                    piece_embeds=next(example_embeds_iter)
                    #consider the last test input
                    if piece_index ==_example_positions[-1] and len(_example_positions)!=1:
                        #last test input use original text
                        #len=1 indicates 0-shot example
                        _reduced_embeds.append(self.get_embedding(piece))
                        length.append(_reduced_embeds[-1].shape[1])
                    else:
                        _reduced_embeds.append(piece_embeds)
                        length.append(piece_embeds.shape[1])

            _reduced_embeds=paddle.concat(_reduced_embeds,axis=1)
            reduced_embeds.append(_reduced_embeds)

            #process label
            _true_positions_t=[]
            for label_index in _label_positions:
                if label_index == -1:
                    #the last token
                    _true_positions_t.append(sum(length)-1) # -1 for causal model
                else:
                    _true_positions_t.append(sum(length[:label_index])-1) # -1 for causal model
            true_positions_t.append(_true_positions_t)
        
        #padding
        inputs_embeds_t,attention_mask_t,true_positions_t=self.padding(instruction_embeddings,reduced_embeds,true_positions_t)


        #get logits
        if mode=="train":
            with paddle.no_grad():
                logits_s=self.get_logits(self.icl_predict(input_ids_s,None,attention_mask_s,label_ids),true_positions_s)

        logits_t=self.get_logits(self.icl_predict(None,inputs_embeds_t,attention_mask_t,label_ids),true_positions_t)

        loss=None
        if mode=="train":
            loss=self.calcu_loss(logits_s,logits_t,label)
        
        return loss,logits_t
        






