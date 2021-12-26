from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, BertTokenizer, RobertaTokenizer, AdamW
from torch import nn
import torch
from transformers import get_linear_schedule_with_warmup
import time
from datetime import timedelta
import datetime as datetime
import random

def give_me_question_from_two_words_(first_word, second_word, before_after_equal_vague_):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    # f'What word occurred before in the text: {first_word} or {second_word}?'
    # 'Which of the following words occurred earlier in time'
    # Which word is earlier in the timeline
    # Does the word Moses occur in the text before the word Isaac?
    # Which word is earlier in the timeline by the text: or ?
    if before_after_equal_vague_ == 'BEFORE':
        return f'Does the word {first_word} occur in the text before the word {second_word}?'
    elif before_after_equal_vague_ == 'AFTER':
        return f'Does the word {first_word} occur in the text after the word {second_word}?'
    elif before_after_equal_vague_ == 'EQUAL':
        return f'Does the word {first_word} and the word {second_word} occur in the text at the same time?'
    elif before_after_equal_vague_ == 'VAGUE':
        return f'Is the temporal relationship in time between the word {first_word} and the word {second_word} unclear?'
def train_TRE_diff_models(model, args, train_dataloader, tokenizer, before_after_equal_vague, percents=None, num_epochs=1):

    print(f'training TRE {before_after_equal_vague}')
    print(f'len of dataloader: {len(train_dataloader)}')
    model.train()
    optim = AdamW(model.parameters(), lr=args.lr)

    if percents:
        percents *= 100
    # the more examples in class, the less weight you should give it
    if before_after_equal_vague == 'EQUAL':
        weights = torch.tensor([100-percents, percents], dtype=torch.float32).to(args.device)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        print(f'EQUAL weights: {weights}')
        criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)

    elif before_after_equal_vague == 'VAGUE':
        weights = torch.tensor([100-percents, percents], dtype=torch.float32).to(args.device)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        print(f'VAGUE weights: {weights}')
        criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights)

    else: # before and after are pretty balanced, so no weights
        criterion = nn.CrossEntropyLoss(reduction='mean')

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)

    print_every = 25
    t = time.time()

    for e in range(1, num_epochs+1, 1):

        LOSS = 0

        for instances_counter, instances in enumerate(train_dataloader, start=1):

            passages = instances[0]
            first_words, second_words, word_labels  = instances[1][0], instances[1][1], instances[1][4]

            for passage, first_word, second_word, Label in zip(passages, first_words, second_words, word_labels):

                # if (Label.strip() != 'VAGUE') & (Label.strip() != 'BEFORE'):
                #     continue

                question = give_me_question_from_two_words_(first_word, second_word, before_after_equal_vague)
                question = question + tokenizer.sep_token

                if Label.strip() == before_after_equal_vague:
                    label = 1
                else:
                    label = 0

                # tokenize question and text as a pair, Roberta
                encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

                input_ids = encodings['input_ids']
                attention_mask = encodings['attention_mask']

                input_ids = torch.tensor([input_ids], requires_grad=False).to(args.device)
                attention_mask = torch.tensor([attention_mask], requires_grad=False).to(args.device)
                labels = torch.tensor([label], requires_grad=False).to(args.device)

                optim.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # print(f'outputs:{outputs}')
                print(torch.softmax(outputs, dim=1))
                print(f'labels:{labels}\n')

                # extract loss
                loss = 0
                loss += criterion(outputs, labels)
                LOSS += loss.item()

                # calculate loss for every parameter that needs grad update
                loss.backward()

                # update parameters
                optim.step()

                # Update the learning rate.
                scheduler.step()

                if instances_counter % print_every == 0:
                    print(f'Epoch:{e}, loss:{round(LOSS, 2)}, Training time:{timedelta(seconds=time.time() - t)},'
                          f' Epoch percent: {round((instances_counter / len(train_dataloader)) * 100, 2)} %\n')
                    LOSS = 0

        torch.save(model.state_dict(), f'C:\\Users\\omerc\\Thesis\\model_tre_Aq_Timebank_{before_after_equal_vague}_epoch_{e}_.pt')
def eval_TRE_diff_models(models_list, args, test_dataloader, tokenizer):

    for model in models_list:
        model.eval()

    print_every = 10
    right, wrong = 0, 0

    for instances_counter, instances in enumerate(test_dataloader, start=1):

        passages = instances[0]
        first_words, second_words, word_labels = instances[1][0], instances[1][1], instances[1][4]

        for passage, first_word, second_word, Label in zip(passages, first_words, second_words, word_labels):

            questions_list = []
            labels_list = ['BEFORE', 'VAGUE', 'EQUAL']
            for i in labels_list:
                question = give_me_question_from_two_words_(first_word, second_word, i)
                question_ = question + tokenizer.sep_token
                questions_list.append(question_)

            results = []
            for model_, question_name, question in zip(models_list, labels_list, questions_list):

                # tokenize question and text as a pair, Roberta
                encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

                input_ids = encodings['input_ids']
                attention_mask = encodings['attention_mask']

                input_ids = torch.tensor([input_ids], requires_grad=False).to(args.device)
                attention_mask = torch.tensor([attention_mask], requires_grad=False).to(args.device)

                with torch.no_grad():
                    outputs = model_(input_ids=input_ids, attention_mask=attention_mask)

                # print(f'outputs:{outputs}')
                # print(f'labels:{labels}\n')
                # print(torch.softmax(outputs, dim=1))
                results.append([question_name, torch.softmax(outputs, dim=1).clone().detach().cpu().numpy()[0],
                                torch.argmax(torch.softmax(outputs, dim=1), dim=1).clone().detach().cpu().numpy()[0]])
                # pred_lables = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            res = 0
            qn_res = 0

            print(results)
            print(Label.strip())
            print("")

            for qn, arr, argmax in results:

                if qn == 'BEFORE':
                    if argmax == 1:
                        qn_res = 'BEFORE'
                        break
                # elif qn == 'AFTER':
                #     if argmax == 1:
                #         qn_res = 'AFTER'
                #         break
                elif qn == 'VAGUE':
                    if argmax == 1:
                        qn_res = 'VAGUE'
                        break
                elif qn == 'EQUAL':
                    if argmax == 1:
                        qn_res = 'EQUAL'

            if qn_res == 0:
                for qn, arr, argmax in results:
                    if arr[1] > res:
                        res = arr[1]
                        qn_res = qn

            if qn_res == Label.strip():
                right += 1
            else:
                wrong += 1

            if instances_counter % print_every == 0:
                print(f'right:{right}')
                print(f'wrong:{wrong}')
                print(f'right / (right + wrong):{right / (right + wrong)}\n')
def eval_TRE_one_model(model, args, test_dataloader, tokenizer, before_after_equal_vague):

    model.eval()
    print_every = 10
    right, wrong = 0, 0

    for instances_counter, instances in enumerate(test_dataloader, start=1):

        passages = instances[0]
        first_words, second_words, word_labels = instances[1][0], instances[1][1], instances[1][4]

        for passage, first_word, second_word, Label in zip(passages, first_words, second_words, word_labels):

            question = give_me_question_from_two_words_(first_word, second_word, before_after_equal_vague)
            question = question + tokenizer.sep_token

            # tokenize question and text as a pair, Roberta
            encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']

            input_ids = torch.tensor([input_ids], requires_grad=False).to(args.device)
            attention_mask = torch.tensor([attention_mask], requires_grad=False).to(args.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)


            print(torch.softmax(outputs, dim=1).clone().detach().cpu().numpy()[0])
            pred_lables = torch.argmax(torch.softmax(outputs, dim=1), dim=1).clone().detach().cpu().numpy()[0]

            if Label.strip() == before_after_equal_vague:
                label = 1
            else:
                label = 0
            # print(f'label:{label}')

            if pred_lables == label:
                right += 1
            else:
                wrong += 1

            if instances_counter % print_every == 0:
                print(f'right:{right}')
                print(f'wrong:{wrong}')
                print(f'right / (right + wrong):{right / (right + wrong)}\n')