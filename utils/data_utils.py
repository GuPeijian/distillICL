from .prompt_utils import make_prompt,make_instruction
import random

def process_input(dataset,tokenizer,batch:list):
    #get instruction
    instruction=make_instruction(dataset)
    instruction_tokens=tokenizer.encode(instruction,return_tensors="pd",return_token_type_ids=False)["input_ids"]

    #get input
    length_label_data=len(dataset.label_data)

    input_tokens=[]
    example_positions=[]
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
        else:
            # (1,..,n/4)-shot example
            max_shot=dataset.n_shot//4
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
        
    return instruction_tokens,input_tokens,example_positions,None,label_positions

def get_label_ids(dataset,tokenizer):
    label_ids=[]
    for label_verb in dataset.id2verb:
        label_verb_token_id = tokenizer.encode(' ' + label_verb,return_token_type_ids=False)["input_ids"][-1]
        label_ids.append(label_verb_token_id)
    return label_ids
