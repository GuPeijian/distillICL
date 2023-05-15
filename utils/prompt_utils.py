"""
1 build instruction
2 make prompt for each input
"""
def make_instruction(dataset):
    dataset_name=dataset.dataset_name
    if dataset_name == 'sst2':
        instruction_fct = instruction_sst2
    return instruction_fct(dataset)


def make_prompt(dataset, ids, mode='train'):
    dataset_name=dataset.dataset_name
    if dataset_name == 'sst2':
        #instruction_fct = instruction_sst2
        template_func = template_sst2
    
    dataset.sample_ids(ids)
    
    #without label
    if mode == 'inference':
        return template_func(dataset.sampled_data[0], None, 'inference')
    
    #with label
    prompt={}
    template=[]
    input_index=[]
    label_index=[]
    for n,ins in enumerate(dataset.sampled_data):
        if n < len(ids)-1:
            _prompt=template_func(ins, dataset.label2verb[ins['label']], 'train')
        else:
            _prompt=template_func(dataset.sampled_data[0], None, 'inference')
        _template=_prompt["template"]
        _input_index=_prompt["input_index"]
        _label_index=_prompt["label_index"]

        #add previous length for idex
        prev_length=len(template)
        template.extend(_template)
        for i,_ in enumerate(_input_index):
            _input_index[i]+=prev_length
        input_index.extend(_input_index)
        if n < len(ids)-1:
            for i,_ in enumerate(_label_index):
                _label_index[i]+=prev_length
        label_index.extend(_label_index)
    
    prompt["template"]=template
    prompt["input_index"]=input_index
    prompt["label_index"]=label_index

    return prompt

def instruction_sst2(dataset):
    instruction="Predict the sentiment of the following sentences. The candidata sentiments are "
    num_label=len(dataset.id2verb)
    for i,label in enumerate(dataset.id2verb):
        if i<num_label-1:
            instruction += f"{label}, "
        else:
            instruction+=f" and {label}.\n"
    return instruction

def template_sst2(ins, label, mode):
    #need to locate each input
    #add _ before ins
    #\n add no _ 
    template=[]
    template.append("Sentence:")
    template.append(f" {ins['sentence']}\n")
    template.append("Sentiment:")
    input_index=[1]
    label_index=[-1] # for inference
    if mode == 'train':
        template.append(f" {label}\n")
        label_index=[3]

    return {"template" : template,
            "input_index" : input_index,
            "label_index" : label_index}
        

