from .prompt_utils import make_prompt,make_instruction,make_prompt_eval
import random

def process_input(dataset,tokenizer,batch:list):
    #get instruction
    instruction=make_instruction(dataset)
    instruction_tokens=tokenizer.encode(instruction,return_tensors="pd",return_token_type_ids=False)["input_ids"]

    #get input
    length_label_data=len(dataset.label_data)

    input_tokens=[]
    example_positions=[]
    label=[] # consider the last label
    label_positions=[]

    for index in batch:
        if index < length_label_data:
            # 0-shot example
            prompt=make_prompt(dataset,[index],mode="inference")
            prompt_tokens=[]
            for piece in prompt["template"]:
                prompt_tokens.append(tokenizer.encode(piece,return_tensors="pd",return_token_type_ids=False)["input_ids"])
            input_tokens.append(prompt_tokens)
            example_positions.append(prompt["input_index"])
            label_positions.append(prompt["label_index"])
            #label
            label_item=dataset.label2id[dataset.label_data[index]["label"]]
            label.append(label_item)
        else:
            # (1,..,n/4)-shot example
            max_shot=dataset.n_shot
            #sample shot
            shot=random.randint(1,max_shot)
            #sample label ids for shot
            prev_length=0
            sample_ids=[]
            for _cls in range(len(dataset.id2verb)):
                sample_ids_cls=random.sample([i+prev_length for i in range(len(dataset.label_data_list[_cls]))],shot)
                sample_ids.extend(sample_ids_cls)
                prev_length+=len(dataset.label_data_list[_cls])
            random.shuffle(sample_ids)
            sample_ids.append(index)
            prompt=make_prompt(dataset,sample_ids,mode="train")
            prompt_tokens=[]
            for piece in prompt["template"]:
                prompt_tokens.append(tokenizer.encode(piece,return_tensors="pd",return_token_type_ids=False)["input_ids"])
            input_tokens.append(prompt_tokens)
            example_positions.append(prompt["input_index"])
            #only consider last loss
            label_positions.append(prompt["label_index"][-1:])
            #label
            label_item=dataset.label2id[dataset.unlabel_data[index-length_label_data]["label"]]
            label.append(label_item)

    return instruction_tokens,input_tokens,example_positions,label,label_positions

def get_label_ids(dataset,tokenizer):
    label_ids=[]
    for label_verb in dataset.id2verb:
        label_verb_token_id = tokenizer.encode(' ' + label_verb,return_token_type_ids=False)["input_ids"][-1]
        label_ids.append(label_verb_token_id)
    return label_ids

def process_input_eval(train_dataset,eval_dataset,tokenizer,batch:list,mode="N"):
    #get instruction
    instruction=make_instruction(train_dataset)
    instruction_tokens=tokenizer.encode(instruction,return_tensors="pd",return_token_type_ids=False)["input_ids"]

    input_tokens=[]
    example_positions=[]
    label_positions=[]
    label=[]

    for index in batch:
        prompt=make_prompt_eval(train_dataset,eval_dataset,index,mode)
        prompt_tokens=[]
        for piece in prompt["template"]:
            prompt_tokens.append(tokenizer.encode(piece,return_tensors="pd",return_token_type_ids=False)["input_ids"])
        input_tokens.append(prompt_tokens)
        example_positions.append(prompt["input_index"])
        label_positions.append(prompt["label_index"])
        label.append(eval_dataset.label2id[eval_dataset[index]["label"]])
    
    return instruction_tokens,input_tokens,example_positions,label,label_positions

def process_input_cl(train_dataset,tokenizer):
    texts=train_dataset.get_texts()
    output_tokens=[]
    for example in texts:
        for item in example:
            item_tokens=tokenizer.encode(item,return_tensors="pd",return_token_type_ids=False)["input_ids"]
            output_tokens.append(item_tokens)
    return output_tokens