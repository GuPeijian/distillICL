import json
import random
from time import sleep
import logging
import argparse
import csv
import os
import time,math
from collections import namedtuple
from tqdm import tqdm

from visualdl import LogWriter

import paddle
import paddle.nn as nn
from paddle.io import DataLoader,TensorDataset

from paddlenlp.transformers import (
    AutoModelForCausalLM,
    GPTLMHeadModel,
    AutoConfig,
    AutoTokenizer,
)

from paddle.optimizer import AdamW
from paddle.optimizer.lr import OneCycleLR
from utils.dataset_utils import SST2Dataset,SUBJDataset
from utils.data_utils import process_input,get_label_ids,process_input_eval,process_input_cl
from utils.model import distillmodel

def set_seed(seed):
     paddle.seed(seed)
     random.seed(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To suppress warnings about parallelism in tokenizers
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/subj/",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="subj",
    )
    parser.add_argument(
        "--llm_dir",
        type=str,
        default="gpt2-xl",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--cl",
        action="store_True",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--N",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # set seeds
    if args.seed is not None:
        set_seed(args.seed)

    #set device
    paddle.device.set_device("gpu:0")

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir)
    # set pad token ids for batched inference cus gpt2 does not have one
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    config={}
    config["llm_path"]=args.llm_dir
    config["CEloss"]=False
    config["alpha"]=args.alpha
    config["beta"]=args.beta
    config["k"]=args.k
    config["pad_token_id"]= tokenizer.pad_token_id
    config["max_length"]=args.max_length

    A=namedtuple("A",config.keys())
    config=A(**config)

    model=distillmodel(config)

    #load dataset
    train_dataset=SUBJDataset(args.data_path,'train')
    #subsample
    train_dataset.subsamplebyshot(n_shot=args.n_shot,N=args.N,seed=args.seed)

    #get label_ids
    label_ids=get_label_ids(train_dataset,tokenizer)
    label_ids=paddle.to_tensor(label_ids)
    
    if args.train:

        num_train=len(train_dataset.label_data) + len(train_dataset.unlabel_data)

        def combine(data_list):
            index_list=[]
            for i in range(len(data_list)):
                index_list.append(data_list[i][0])
            return paddle.concat(index_list)
        
        index_dataset=TensorDataset([paddle.to_tensor([i for i in range(num_train)]).unsqueeze(1).cpu()])

        train_dataloader=DataLoader(index_dataset,batch_size=args.train_batch_size,shuffle=True,collate_fn=combine,drop_last=False)

        model.train()

        #freeze LLM
        for n, p in model.named_parameters():
            if "LLM" in n:
                p.trainable = False

        #optimizer
        max_train_steps = args.num_train_epochs * len(train_dataloader)
        if args.cl:
            max_train_steps+=args.num_train_epochs
        lr_scheduler = OneCycleLR(
            max_learning_rate=args.learning_rate,
            phase_pct=args.warmup_ratio,
            end_learning_rate=0.0,
            total_steps=max_train_steps,
            anneal_strategy="linear",
        )
        #parameters
        no_update=["LLM"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_update)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(learning_rate=lr_scheduler,parameters=optimizer_grouped_parameters)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_train}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total optimization steps = {max_train_steps}")

        writer = LogWriter(os.getenv("VDL_LOG_PATH"))
        completed_steps = 0
        time_log = time.time()
        total_loss=0.0
        last_loss=0.0

        #prepare n shot example for cl
        if args.cl:
            n_shot_tokens=process_input_cl(train_dataset,tokenizer)

        for epoch in range(args.num_train_epochs):
            if args.cl:
                loss=model.constructive_reduce(n_shot_tokens)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                completed_steps+=1
                loss=float(loss.cpu().numpy())
                logger.info(f"steps: {completed_steps}/{max_train_steps},"
                            f" epoch: {epoch}/{args.num_train_epochs},"
                            f" lr: {lr_scheduler.get_lr():.2e},"
                            f" cl_loss: {loss},")
                writer.add_scalar('cl_loss', loss, epoch+1)
                writer.add_scalar('lr', lr_scheduler.get_lr(), completed_steps)

            for step,batch in enumerate(train_dataloader):
                batch=batch.tolist()
                #process each item
                instruction_tokens,input_tokens,example_positions,label,label_positions=process_input(train_dataset,tokenizer,batch)
                
                if label is not None:
                    # to tensor
                    label=paddle.to_tensor(label) #[N]

                loss,_=model(instruction_tokens=instruction_tokens,
                        input_tokens=input_tokens,
                        example_positions=example_positions,
                        label_ids=label_ids,
                        label=label,
                        label_positions=label_positions)
                
                # optimize
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_gradients()
                completed_steps += 1
                
                #log loss
                loss=float(loss.cpu().numpy())
                writer.add_scalar('loss', loss, completed_steps)
                writer.add_scalar('lr', lr_scheduler.get_lr(), completed_steps)
                total_loss+=loss

                if completed_steps % 10 == 0:
                    logger.info(f"steps: {completed_steps}/{max_train_steps},"
                                f" epoch: {epoch}/{args.num_train_epochs},"
                                f" lr: {lr_scheduler.get_lr():.2e},"
                                f" loss: {(total_loss-last_loss)/10},"
                                f" efficiency: {10 / (time.time() - time_log):.2f}steps/s")
                    time_log = time.time()
                    last_loss=total_loss

    #eval
    test_dataset=SUBJDataset(args.data_path,'test')
    model.eval()

    #shuffle label data
    random.shuffle(train_dataset.label_data)
    #shuffle data
    random.shuffle(train_dataset.data)
    index=[i for i in range(len(test_dataset))]

    num_iter=math.ceil(len(test_dataset)/args.eval_batch_size)

    num_tp_N=0
    num_tp_n_shot=0

    for i in tqdm(range(num_iter)):
        batch=index[i*args.eval_batch_size:(i+1)*args.eval_batch_size]
        instruction_tokens,input_tokens,example_positions,label,label_positions=process_input_eval(train_dataset,test_dataset,tokenizer,batch,"N")
        with paddle.no_grad():
            _,logits=model(instruction_tokens=instruction_tokens,
                        input_tokens=input_tokens,
                        example_positions=example_positions,
                        label_ids=label_ids,
                        label=label,
                        label_positions=label_positions,
                        mode="eval")
        
        assert logits.shape[0]==len(label)

        preds=paddle.argmax(logits,axis=1).cpu().tolist()

        for n,pred in enumerate(preds):
            if pred == label[n]:
                num_tp_N+=1


    for i in tqdm(range(num_iter)):
        batch=index[i*args.eval_batch_size:(i+1)*args.eval_batch_size]
        instruction_tokens,input_tokens,example_positions,label,label_positions=process_input_eval(train_dataset,test_dataset,tokenizer,batch,"n_shot")
        with paddle.no_grad():
            _,logits=model(instruction_tokens=instruction_tokens,
                        input_tokens=input_tokens,
                        example_positions=example_positions,
                        label_ids=label_ids,
                        label=label,
                        label_positions=label_positions,
                        mode="eval")
        
        assert logits.shape[0]==len(label)

        preds=paddle.argmax(logits,axis=1).cpu().tolist()

        for n,pred in enumerate(preds):
            if pred == label[n]:
                num_tp_n_shot+=1

    acc_N=num_tp_N/len(test_dataset)

    acc_n_shot=num_tp_n_shot/len(test_dataset)

    save_results_file = os.path.join(args.output_dir, 'results.csv')
    csv_exists = os.path.isfile(save_results_file)
    with open(save_results_file, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not csv_exists:
            csvwriter.writerow(['dataset','seed','N','n_shot','k','cl','epoch','bs','lr','alpha','beta','acc_N','acc_n_shot'])
        csvwriter.writerow([args.dataset_name,
                            args.seed,
                            args.N,
                            args.n_shot,
                            args.k,
                            args.cl,
                            args.num_train_epochs,
                            args.train_batch_size,
                            args.learning_rate,
                            args.alpha,
                            args.beta,
                            acc_N,
                            acc_n_shot
                            ])



if __name__=="__main__":
    main()