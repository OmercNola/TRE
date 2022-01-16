from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, BertTokenizer, RobertaTokenizer, AdamW
from torch import nn
import torch
from transformers import get_linear_schedule_with_warmup
import time
from datetime import timedelta
import datetime as datetime
import random
from pathlib import Path
"============================================================================="
# without markers:
def before_question_from_two_words(first_word, second_word):
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
    return f'Did the word {first_word} occur in the timeline of the text before the word {second_word}?'
def after_question_from_two_words(first_word, second_word):
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
    return f'Did the word {first_word} occur in the timeline of the text after the word {second_word}?'
def equal_question_from_two_words(first_word, second_word):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    return f'Did the word {first_word} and the word {second_word} occur in the timeline of the text at the same time?'
def vague_question_from_two_words(first_word, second_word):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    return  f'Is it not possible to know if the word {first_word} occurred in the timeline of the text before the word {second_word}?'
def train_TRE(model, args, train_dataloader, tokenizer, num_epochs=1):

    print('training TRE')
    model.train()
    optim = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=500, num_training_steps=total_steps)

    print_every = 50
    t = time.time()

    for e in range(1, num_epochs+1, 1):

        LOSS = 0

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for instances_counter, instances in enumerate(train_dataloader, start=1):

            passages = instances[0]
            first_words, second_words, word_labels  = instances[1][0], instances[1][1], instances[1][4]

            for passage, first_word, second_word, Label in zip(passages, first_words, second_words, word_labels):

                if (Label.strip() == 'EQUAL'):
                    continue

                question = before_question_from_two_words(first_word, second_word)
                question_before = question + tokenizer.sep_token
                question = after_question_from_two_words(first_word, second_word)
                question_after = question + tokenizer.sep_token
                question = equal_question_from_two_words(first_word, second_word)
                question_equal = question + tokenizer.sep_token
                question = vague_question_from_two_words(first_word, second_word)
                question_vague = question + tokenizer.sep_token

                # if (Label.strip() == 'EQUAL'):
                #     questions_list = [question_before, question_after, question_equal, question_vague]
                #     label_list = ['BEFORE', 'AFTER', 'EQUAL', 'VAGUE']

                # if (Label.strip() == 'VAGUE'):
                #     rand = random.random()
                #     if rand <= 0.3:  # 30%
                #         questions_list = [question_before, question_after, question_equal, question_vague]
                #         label_list = ['BEFORE', 'AFTER', 'VAGUE']
                #     else:
                #         questions_list = [question_before, question_after, question_vague]
                #         label_list = ['BEFORE', 'AFTER', 'VAGUE']

                # elif (Label.strip() == 'BEFORE') | (Label.strip() == 'AFTER'):
                #     rand = random.random()
                #     if rand <= 0.8: # 80%
                #         questions_list = [question_before, question_after]
                #         label_list = ['BEFORE', 'AFTER']
                #     elif 0.8 < rand <= 0.90: # 10%
                #         questions_list = [question_before, question_after, question_vague]
                #         label_list = ['BEFORE', 'AFTER', 'VAGUE']
                #     else: # 10%
                #         questions_list = [question_before, question_after, question_equal, question_vague]
                #         label_list = ['BEFORE', 'AFTER', 'EQUAL', 'VAGUE']

                questions_list = [question_before, question_after, question_vague]
                label_list = ['BEFORE', 'AFTER', 'VAGUE']

                for question_name, question in zip(label_list, questions_list):

                    # print(f'label:{Label.strip()}')
                    # print(f'question_name:{question_name}')

                    if question_name == Label.strip():
                        label = 1
                    else:
                        label = 0

                    # tokenize question and text as a pair, Roberta
                    encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

                    input_ids = encodings['input_ids']
                    attention_mask = encodings['attention_mask']

                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    batch_labels.append(label)

                    if len(batch_input_ids) == args.batch_size:

                        batch_input_ids = torch.tensor(batch_input_ids, requires_grad=False).to(args.device)
                        batch_attention_mask = torch.tensor(batch_attention_mask, requires_grad=False).to(args.device)
                        batch_labels = torch.tensor(batch_labels, requires_grad=False).to(args.device)

                        optim.zero_grad()

                        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

                        # print(f'outputs:{outputs}')
                        # print(torch.softmax(outputs, dim=1))
                        # print(f'labels:{batch_labels}\n')

                        # extract loss
                        loss = 0
                        loss += criterion(outputs, batch_labels)
                        LOSS += loss.item()

                        # calculate loss for every parameter that needs grad update
                        loss.backward()

                        # This is to help prevent the "exploding gradients" problem.
                        # torch.nn.utils.clip_grad_norm_((i for i in model.parameters() if i.requires_grad == True), 40)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                        # update parameters
                        optim.step()

                        # Update the learning rate.
                        scheduler.step()

                        batch_input_ids = []
                        batch_attention_mask = []
                        batch_labels = []

            if instances_counter % print_every == 0:
                print(f'Epoch:{e}, loss:{round(LOSS, 2)}, Training time:{timedelta(seconds=time.time() - t)},'
                      f' Epoch percent: {round((instances_counter / len(train_dataloader)) * 100, 2)} %\n')
                LOSS = 0

        torch.save(model.state_dict(), Path(f'models/model_TRE_Aq_Timebank_epoch_{e}_.pt'))
def eval_TRE(model, args, test_dataloader, tokenizer):

    model.eval()
    print_every = 10
    right, wrong = 0, 0

    for instances_counter, instances in enumerate(test_dataloader, start=1):

        passages = instances[0]
        first_words, second_words, word_labels = instances[1][0], instances[1][1], instances[1][4]

        for passage, first_word, second_word, Label in zip(passages, first_words, second_words, word_labels):

            if Label.strip() == 'EQUAL':
                continue
            #
            # print(Label)

            question = before_question_from_two_words(first_word, second_word)
            question_before = question + tokenizer.sep_token
            question = after_question_from_two_words(first_word, second_word)
            question_after = question + tokenizer.sep_token
            question = equal_question_from_two_words(first_word, second_word)
            question_equal = question + tokenizer.sep_token
            question = vague_question_from_two_words(first_word, second_word)
            question_vague = question + tokenizer.sep_token
            questions_list = [question_before, question_after, question_vague]
            label_list = ['BEFORE', 'AFTER', 'VAGUE']

            results = []

            for question_name, question in zip(label_list, questions_list):

                # if question_name != 'VAGUE':
                #     continue

                # print(question_name)
                # print(question)

                # tokenize question and text as a pair, Roberta
                encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

                input_ids = encodings['input_ids']
                attention_mask = encodings['attention_mask']

                input_ids = torch.tensor([input_ids], requires_grad=False).to(args.device)
                attention_mask = torch.tensor([attention_mask], requires_grad=False).to(args.device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # print(f'outputs:{outputs}')
                # print(f'labels:{labels}\n')
                # print(torch.softmax(outputs, dim=1))
                results.append([question_name, torch.softmax(outputs, dim=1).clone().detach().cpu().numpy()[0],
                                torch.argmax(torch.softmax(outputs, dim=1), dim=1).clone().detach().cpu().numpy()[0]])
                # pred_lables = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            res = 0
            qn_res = 0

            # print(results)
            # print(Label.strip())
            # print("")

            for qn, arr, argmax in results:
                if argmax == 1:
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
"============================================================================="
# for markers:
def before_question_for_markers(first_word, second_word):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    return f'Did the entity [E1] {first_word} [/E1] occur in the timeline of the text before the entity [E2] {second_word} [/E2]?'
def after_question_for_markers(first_word, second_word):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    return f'Did the entity [E1] {first_word} [/E1] occur in the timeline of the text after the entity [E2] {second_word} [/E2]?'
def equal_question_for_markers(first_word, second_word):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    return f'Does [E1] and [E2] occur in the text at the same time?'
def vague_question_for_markers(first_word, second_word):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    f'Is it difficult to compare the entities [E1] {first_word} [/E1] and [E2] {second_word} [/E2] in the timeline of the text?'
    return  f'Is it not possible to know if the entity [E1] {first_word} [/E1] occurred in the timeline of the text before the entity [E2] {second_word} [/E2]?'
def train_TRE_with_markers(
        model, args, train_dataloader,
        tokenizer, num_epochs=1):

    print('training TRE with markers')
    model.train()
    optim = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)

    print_every = 50
    t = time.time()

    for e in range(1, num_epochs+1, 1):

        LOSS = 0

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for instances_counter, instances in enumerate(train_dataloader, start=1):

            passages = instances[0]
            first_words, second_words, word_labels  = instances[1][0], instances[1][1], instances[1][4]

            for passage, first_word, second_word, Label in zip(passages, first_words, second_words, word_labels):
                if (Label.strip() == 'EQUAL'):
                    continue

                question_before = before_question_for_markers(first_word, second_word) + tokenizer.sep_token
                question_after = after_question_for_markers(first_word, second_word) + tokenizer.sep_token
                # question_equal = equal_question_for_markers(first_word, second_word) + tokenizer.sep_token
                question_vague = vague_question_for_markers(first_word, second_word) + tokenizer.sep_token


                questions_list = [question_before, question_after, question_vague]
                label_list = ['BEFORE', 'AFTER', 'VAGUE']

                for question_name, question in zip(label_list, questions_list):

                    if question_name == Label.strip():
                        label = 1
                    else:
                        label = 0

                    # tokenize question and text as a pair, Roberta
                    encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

                    input_ids = encodings['input_ids']
                    attention_mask = encodings['attention_mask']

                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    batch_labels.append(label)

                    if len(batch_input_ids) == args.batch_size:

                        batch_input_ids = torch.tensor(batch_input_ids, requires_grad=False).to(args.device)
                        batch_attention_mask = torch.tensor(batch_attention_mask, requires_grad=False).to(args.device)
                        batch_labels = torch.tensor(batch_labels, requires_grad=False).to(args.device)

                        optim.zero_grad()

                        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

                        # print(f'outputs:{outputs}')
                        # print(torch.softmax(outputs, dim=1))
                        # print(f'labels:{batch_labels}\n')

                        # extract loss
                        loss = 0
                        loss += criterion(outputs, batch_labels)
                        LOSS += loss.item()

                        # calculate loss for every parameter that needs grad update
                        loss.backward()

                        # This is to help prevent the "exploding gradients" problem.
                        # torch.nn.utils.clip_grad_norm_((i for i in model.parameters() if i.requires_grad == True), 40)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                        # update parameters
                        optim.step()

                        # Update the learning rate.
                        scheduler.step()

                        batch_input_ids = []
                        batch_attention_mask = []
                        batch_labels = []

            if instances_counter % print_every == 0:
                print(f'Epoch:{e}, loss:{round(LOSS, 2)}, Training time:{timedelta(seconds=time.time() - t)},'
                      f' Epoch percent: {round((instances_counter / len(train_dataloader)) * 100, 2)} %\n')
                LOSS = 0

        torch.save(model.state_dict(), Path(f'models/model_with_markers_Aq_Timebank_Before_after_vague_epoch_{6+e}_.pt'))
def eval_TRE_with_markers(model, args, test_dataloader, tokenizer):
    model.eval()
    print_every = 10
    right, wrong = 0, 0

    for instances_counter, instances in enumerate(test_dataloader, start=1):

        passages = instances[0]
        first_words, second_words, word_labels = instances[1][0], instances[1][1], instances[1][4]

        for passage, first_word, second_word, Label in zip(passages, first_words, second_words, word_labels):


            question_before = before_question_for_markers(first_word, second_word) + tokenizer.sep_token
            question_after = after_question_for_markers(first_word, second_word) + tokenizer.sep_token
            question_equal = equal_question_for_markers(first_word, second_word) + tokenizer.sep_token
            question_vague = vague_question_for_markers(first_word, second_word) + tokenizer.sep_token
            questions_list = [question_before, question_after, question_vague]
            label_list = ['BEFORE', 'AFTER', 'VAGUE']

            results = []

            for question_name, question in zip(label_list, questions_list):
                # if question_name != 'VAGUE':
                #     continue

                # print(question_name)
                # print(question)

                # tokenize question and text as a pair, Roberta
                encodings = tokenizer(question, passage, max_length=args.Max_Len, padding='max_length', truncation=True)

                input_ids = encodings['input_ids']
                attention_mask = encodings['attention_mask']

                input_ids = torch.tensor([input_ids], requires_grad=False).to(args.device)
                attention_mask = torch.tensor([attention_mask], requires_grad=False).to(args.device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # print(f'outputs:{outputs}')
                # print(f'labels:{labels}\n')
                # print(torch.softmax(outputs, dim=1))
                results.append([question_name, torch.softmax(outputs, dim=1).clone().detach().cpu().numpy()[0],
                                torch.argmax(torch.softmax(outputs, dim=1), dim=1).clone().detach().cpu().numpy()[0]])
                # pred_lables = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            res = 0
            qn_res = 0

            # print(results)
            # print(Label.strip())
            # print("")

            for qn, arr, argmax in results:
                if argmax == 1:
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
"============================================================================="
# New questions for markers:
class results_tracker:

    def __init__(
            self,
            total_amount_of_before_instances,
            total_amount_of_after_instances,
            total_amount_of_vague_instances,
            total_amount_of_equal_instances
    ):

        self.total_amount_of_before_instances = \
            total_amount_of_before_instances
        self.total_amount_of_after_instances = \
            total_amount_of_after_instances
        self.total_amount_of_vague_instances = \
            total_amount_of_vague_instances
        self.total_amount_of_equal_instances = \
            total_amount_of_equal_instances


        self.TP_BEFORE = 0
        self.TN_BEFORE = 0
        self.FP_BEFORE = 0
        self.FN_BEFORE = 0

        self.TP_AFTER = 0
        self.TN_AFTER = 0
        self.FP_AFTER = 0
        self.FN_AFTER = 0

        self.TP_EQUAL = 0
        self.TN_EQUAL = 0
        self.FP_EQUAL = 0
        self.FN_EQUAL = 0

        self.TP_VAGUE = 0
        self.TN_VAGUE = 0
        self.FP_VAGUE = 0
        self.FN_VAGUE = 0

    def update(self, label, ans1, ans2):

        if label.strip() == 'BEFORE':

            if ans1 == 1 and ans2 == 0:
                self.TP_BEFORE += 1

            if ans1 == 0 and ans2 == 1:
                self.FP_AFTER += 1

            if ans1 == 1 and ans2 == 1:
                self.FP_VAGUE += 1

            if ans1 == 0 and ans2 == 0:
                self.FP_EQUAL += 1
def question_1_for_markers(first_word, second_word):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    res = f'Is it possible that the start time of entity [E1] {first_word} [/E1]' \
          f' is before the start time of entity [E2] {second_word} [/E2]' \
          f' in the timeline of the text?'
    return res
def question_2_for_markers(first_word, second_word):
    """
    :param first_word:
    :param second_word:
    :return:
    """
    res = f'Is it possible that the start time of entity [E2] {second_word} [/E2]' \
          f' is before the start time of entity [E1] {first_word} [/E1]' \
          f' in the timeline of the text?'
    return res
def get_label(question_name, label):
    """
    :param question_name:
    :type question_name:
    :param label:
    :type label:
    :return:
    :rtype:
    """
    if question_name == 'question_1':
        if label.strip() == 'BEFORE':
            res = 1
        elif label.strip() == 'AFTER':
            res = 0
        elif label.strip() == 'VAGUE':
            res = 1
        elif label.strip() == 'EQUAL':
            res = 0

    elif question_name == 'question_2':
        if label.strip() == 'BEFORE':
            res = 0
        elif label.strip() == 'AFTER':
            res = 1
        elif label.strip() == 'VAGUE':
            res = 1
        elif label.strip() == 'EQUAL':
            res = 0

    return res
def check_answers(ans1, ans2, Label):
    """
    :param ans1:
    :type ans1:
    :param ans2:
    :type ans2:
    :param Label:
    :type Label:
    :return:
    :rtype:
    """

    if ans1 == 0 and ans2 == 0:
        if Label.strip() == 'EQUAL':
            res = True
        else:
            res = False

    elif ans1 == 1 and ans2 == 1:
        if Label.strip() == 'VAGUE':
            res = True
        else:
            res = False

    elif ans1 == 1 and ans2 == 0:
        if Label.strip() == 'BEFORE':
            res = True
        else:
            res = False

    elif ans1 == 0 and ans2 == 1:
        if Label.strip() == 'AFTER':
            res = True
        else:
            res = False

    return res
def print_training_progress(
        t, length_of_data_loader, epoch, instances_counter, total_loss):
    """
    """
    print(f'Epoch:{epoch}, '
          f' loss:{total_loss:.2f}, '
          f'Training time:{timedelta(seconds=time.time() - t)}, '
          f'Epoch percent: {round((instances_counter / length_of_data_loader) * 100, 2)}')
def train_tre_new_questions_with_markers(
        model, args, train_dataloader,
        tokenizer, num_epochs=1):
    """
    :param model:
    :type model:
    :param args:
    :type args:
    :param train_dataloader:
    :type train_dataloader:
    :param tokenizer:
    :type tokenizer:
    :param num_epochs:
    :type num_epochs:
    :return:
    :rtype:
    """

    print('training tre with markers...')

    model.train()

    optim = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=total_steps
    )

    print_every = 50
    t = time.time()

    for e in range(1, num_epochs+1, 1):

        LOSS = 0

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for instances_counter, instances in enumerate(train_dataloader, start=1):

            passages = instances[0]
            first_words = instances[1][0]
            second_words = instances[1][1]
            word_labels = instances[1][4]

            zip_object = zip(passages, first_words, second_words, word_labels)
            for passage, first_word, second_word, Label in zip_object:

                question_1 = question_1_for_markers(
                    first_word, second_word) + tokenizer.sep_token
                question_2 = question_2_for_markers(
                    first_word, second_word) + tokenizer.sep_token

                questions_list = [
                    ('question_1', question_1),
                    ('question_2', question_2)
                ]

                for question_name, question in questions_list:

                    label = get_label(question_name, Label)

                    # tokenize question and text as a pair, Roberta
                    encodings = tokenizer(
                        question,
                        passage,
                        max_length=args.Max_Len,
                        padding='max_length',
                        truncation=True
                    )

                    input_ids = encodings['input_ids']
                    attention_mask = encodings['attention_mask']

                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    batch_labels.append(label)

                    # compute loss and update weights every args.batch_size:
                    if len(batch_input_ids) == args.batch_size:

                        batch_input_ids = torch.tensor(
                            batch_input_ids, requires_grad=False).to(args.device)
                        batch_attention_mask = torch.tensor(
                            batch_attention_mask, requires_grad=False).to(args.device)
                        batch_labels = torch.tensor(
                            batch_labels, requires_grad=False).to(args.device)

                        # zero gradients before update:
                        optim.zero_grad()

                        # forward pass:
                        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

                        # extract loss
                        loss = 0
                        loss += criterion(outputs, batch_labels)
                        LOSS += loss.item()

                        # calculate loss
                        loss.backward()

                        # This is to help prevent the "exploding gradients" problem:
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

                        # update parameters
                        optim.step()

                        # Update the learning rate.
                        scheduler.step()

                        batch_input_ids = []
                        batch_attention_mask = []
                        batch_labels = []

            if instances_counter % print_every == 0:
                print_training_progress(
                    t, len(train_dataloader), e, instances_counter, LOSS
                )
                LOSS = 0

        torch.save(model.state_dict(), Path(f'models/model_epoch_{e}_.pt'))
def eval_tre_new_questions_with_markers(
        model, args, test_dataloader, tokenizer):
    """
    :param model:
    :type model:
    :param args:
    :type args:
    :param test_dataloader:
    :type test_dataloader:
    :param tokenizer:
    :type tokenizer:
    :return:
    :rtype:
    """

    model.eval()
    print_every = 10
    right, wrong = 0, 0

    for instances_counter, instances in enumerate(test_dataloader, start=1):

        passages = instances[0]
        first_words, second_words, word_labels = instances[1][0], instances[1][1], instances[1][4]

        zip_object = zip(passages, first_words, second_words, word_labels)
        for passage, first_word, second_word, Label in zip_object:

            question_1 = question_1_for_markers(
                first_word, second_word) + tokenizer.sep_token
            question_2 = question_2_for_markers(
                first_word, second_word) + tokenizer.sep_token

            questions_list = [
                ('question_1', question_1),
                ('question_2', question_2)
            ]

            # 2 Questions for each instance:
            results = []
            for question_name, question in questions_list:

                # tokenize question and text as a pair, Roberta
                encodings = tokenizer(
                    question,
                    passage,
                    max_length=args.Max_Len,
                    padding='max_length',
                    truncation=True
                )

                input_ids = encodings['input_ids']
                attention_mask = encodings['attention_mask']

                input_ids = torch.tensor(
                    [input_ids], requires_grad=False).to(args.device)
                attention_mask = torch.tensor(
                    [attention_mask], requires_grad=False).to(args.device)

                with torch.no_grad():

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    pred = torch.argmax(
                        torch.softmax(outputs, dim=1),
                        dim=1
                    ).clone().detach().cpu().numpy()[0]
                    results.append([question_name, pred])

            ans1, ans2 = results[0][1], results[1][1]

            if check_answers(ans1, ans2, Label):
                right += 1
                TP += 1
            else:
                wrong += 1
                FN += 1

        if instances_counter % print_every == 0:

            # precision and recall:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            F1 = 2 * (precision * recall) / (precision + recall)
            print(f'F1: {F1}')

            print(f'right:{right}')
            print(f'wrong:{wrong}')
            print(f'Accuracy): {right / (right + wrong)}\n')
"============================================================================="


