from __future__ import absolute_import, division, print_function
from transformers import (AutoTokenizer, AutoModel, AutoModelForQuestionAnswering,
                          BertTokenizer, RobertaTokenizer, AdamW)
from torch import nn
import torch
from transformers import get_linear_schedule_with_warmup
import time
from datetime import timedelta
import datetime as datetime
import random
from pathlib import Path
# BOOLQ dataset
def train_boolq(model, args, train_dataloader, tokenizer, num_epochs=1):
    print('training..')
    model.train()
    optim = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=500, num_training_steps=total_steps
    )

    print_every = 25
    t = time.time()

    for e in range(1, num_epochs+1, 1):

        LOSS = 0

        for instances_counter, instances in enumerate(train_dataloader, start=1):

            batch_input_ids = []
            batch_attention_mask = []
            labels = []

            questions = instances['question']
            answers = instances['answer']
            passages = instances['passage']

            for question, answer, passage in zip(questions, answers, passages):

                question = question + tokenizer.sep_token
                assert answer == True or answer == False

                label = 1 if answer == True else 0
                labels.append(label)

                # tokenize question and text as a pair, Roberta
                encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

                input_ids = encodings['input_ids']
                attention_mask = encodings['attention_mask']

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)

            batch_input_ids = torch.tensor(batch_input_ids, requires_grad=False).to(args.device)
            batch_attention_mask = torch.tensor(batch_attention_mask, requires_grad=False).to(args.device)
            labels = torch.tensor(labels, requires_grad=False).to(args.device)

            optim.zero_grad()

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

            # extract loss
            loss = 0
            loss += criterion(outputs, labels)
            LOSS += loss.item()

            if instances_counter % print_every == 0:
                print(f'Epoch:{e}, loss:{round(LOSS, 2)}, Training time:{timedelta(seconds=time.time() - t)},'
                      f' Epoch percent: {round((instances_counter / len(train_dataloader)) * 100, 2)} %')
                LOSS = 0

            # calculate loss for every parameter that needs grad update
            loss.backward()

            # This is to help prevent the "exploding gradients" problem.
            # torch.nn.utils.clip_grad_norm_((i for i in model.parameters() if i.requires_grad == True), 40)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # update parameters
            optim.step()

            # Update the learning rate.
            scheduler.step()

        torch.save(model.state_dict(), Path(f'models/model_boolq_with_markers_epoch_{e}_.pt'))
def eval_boolq(model, args, test_dataloader, tokenizer):

    model.eval()

    print_every = 10
    right, wrong = 0, 0

    for instances_counter, instances in enumerate(test_dataloader, start=1):

        batch_input_ids = []
        batch_attention_mask = []
        labels = []

        questions = instances['question']
        answers = instances['answer']
        passages = instances['passage']

        for question, answer, passage in zip(questions, answers, passages):
            question = question + tokenizer.sep_token
            assert answer == True or answer == False

            label = 1 if answer == True else 0
            labels.append(label)

            # tokenize question and text as a pair, Roberta
            encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        batch_input_ids = torch.tensor(batch_input_ids, requires_grad=False).to(args.device)
        batch_attention_mask = torch.tensor(batch_attention_mask, requires_grad=False).to(args.device)
        labels = torch.tensor(labels, requires_grad=False).to(args.device)

        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

        pred_lables = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        for label, pred_lable in zip(labels, pred_lables):
            if label == pred_lable:
                right += 1
            else:
                wrong += 1

        # copmute accuracy:
        if instances_counter % print_every == 0:
            print(f'acc): {right / (right + wrong)}\n')
"================================================================================="
"==============================  BOOLQ WITH MARKERS  ============================="
"================================================================================="
# # # Datasets:
# dataset_boolq = load_dataset("boolq")
# # Dataloaders:
# train_dataloader = DataLoader(dataset_boolq['train'], batch_size=args.batch_size, shuffle=True)
#
# # # Training
# #train_boolq(model, args, train_dataloader, tokenizer, num_epochs=10)
#
# # Evaluation
# test_dataloader = DataLoader(dataset_boolq['validation'], batch_size=4, shuffle=False)
# PATH = Path('models/model_boolq_with_markers_epoch_10_.pt')
# model.load_state_dict(torch.load(PATH))
# model.to(args.device)
# eval_boolq(model, args, test_dataloader, tokenizer)