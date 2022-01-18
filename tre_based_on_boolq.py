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
from sklearn.metrics import f1_score
"============================================================================="
# New questions for markers:
class results_tracker:

    def __init__(self):

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

            if ans1 == 1 and ans2 == 0: # BEFORE
                self.TP_BEFORE += 1

            if ans1 == 0 and ans2 == 1: # AFTER
                self.FP_AFTER += 1
                self.FN_BEFORE += 1

            if ans1 == 1 and ans2 == 1: # VAGUE
                self.FP_VAGUE += 1
                self.FN_BEFORE += 1

            if ans1 == 0 and ans2 == 0: # EQUAL
                self.FP_EQUAL += 1
                self.FN_BEFORE += 1

        elif label.strip() == 'AFTER':

            if ans1 == 1 and ans2 == 0: # BEFORE
                self.FP_BEFORE += 1
                self.FN_AFTER += 1

            if ans1 == 0 and ans2 == 1: # AFTER
                self.TP_AFTER += 1

            if ans1 == 1 and ans2 == 1: # VAGUE
                self.FP_VAGUE += 1
                self.FN_AFTER += 1

            if ans1 == 0 and ans2 == 0: # EQUAL
                self.FP_EQUAL += 1
                self.FN_AFTER += 1

        elif label.strip() == 'EQUAL':

            if ans1 == 1 and ans2 == 0: # BEFORE
                self.FP_BEFORE += 1
                self.FN_EQUAL += 1

            if ans1 == 0 and ans2 == 1: # AFTER
                self.FP_AFTER += 1
                self.FN_EQUAL += 1

            if ans1 == 1 and ans2 == 1: # VAGUE
                self.FP_VAGUE += 1
                self.FN_EQUAL += 1

            if ans1 == 0 and ans2 == 0: # EQUAL
                self.TP_EQUAL += 1

    def f1_macro_and_micro(self):
        """
        F1-score = 2 × (precision × recall)/(precision + recall)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)

        F1_micro = 2 × (micro_precision × micro_recall)/(micro_precision + micro_recall)
        micro_precision = TP_sum_all_classes/(TP_sum_all_classes + FP_sum_all_classes)
        micro_recall = TP_sum_all_classes/(TP_sum_all_classes + FN_sum_all_classes)
        """

        "====================================================================================="
        "BEFORE"
        try:
            precision_before = self.TP_BEFORE / (self.TP_BEFORE + self.FP_BEFORE)
        except ZeroDivisionError:
            precision_before = 0
        try:
            recall_before = self.TP_BEFORE / (self.TP_BEFORE + self.FN_BEFORE)
        except ZeroDivisionError:
            recall_before = 0
        try:
            f1_before = 2 * (precision_before * recall_before) / (precision_before + recall_before)
        except ZeroDivisionError:
            f1_before = 0
        "====================================================================================="
        "AFTER"
        try:
            precision_after = self.TP_AFTER / (self.TP_AFTER + self.FP_AFTER)
        except ZeroDivisionError:
            precision_after = 0

        try:
            recall_after = self.TP_AFTER / (self.TP_AFTER + self.FN_AFTER)
        except ZeroDivisionError:
            recall_after = 0

        try:
            f1_after = 2 * (precision_after * recall_after) / (precision_after + recall_after)
        except ZeroDivisionError:
            f1_after = 0
        "====================================================================================="
        "EQUAL"
        try:
            precision_equal = self.TP_EQUAL / (self.TP_EQUAL + self.FP_EQUAL)
        except ZeroDivisionError:
            precision_equal = 0

        try:
            recall_equal = self.TP_EQUAL / (self.TP_EQUAL + self.FN_EQUAL)
        except ZeroDivisionError:
            recall_equal = 0

        try:
            f1_equal = 2 * (precision_equal * recall_equal) / (precision_equal + recall_equal)
        except ZeroDivisionError:
            f1_equal= 0
        "====================================================================================="
        "F1, MACRO, MICRO"

        # macro f1, just the everage:
        macro_f1 = (f1_before + f1_after + f1_equal) / 3

        # micro f1
        TP_sum_all_classes = self.TP_BEFORE + self.TP_AFTER + self.TP_EQUAL
        FP_sum_all_classes = self.FP_BEFORE + self.FP_AFTER + self.FP_EQUAL
        FN_sum_all_classes = self.FN_BEFORE + self.FN_AFTER + self.FN_EQUAL

        try:
            micro_precision = TP_sum_all_classes / (TP_sum_all_classes + FP_sum_all_classes)
        except ZeroDivisionError:
            micro_precision = 0
        try:
            micro_recall = TP_sum_all_classes/(TP_sum_all_classes + FN_sum_all_classes)
        except ZeroDivisionError:
            micro_recall = 0
        try:
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        except ZeroDivisionError:
            micro_f1 = 0

        return (float(f'{macro_f1:.4f}'), float(f'{micro_f1:.4f}'))
def question_1_for_markers(first_word, second_word):
    """
    :param first_word:
    :type first_word:
    :param second_word:
    :type second_word:
    :return:
    :rtype:
    """
    res = f'Is it possible that [E1] {first_word} [/E1] started before [E2] {second_word} [/E2]?'
    return res
def question_2_for_markers(first_word, second_word):
    """
    :param first_word:
    :type first_word:
    :param second_word:
    :type second_word:
    :return:
    :rtype:
    """
    res = f'Is it possible that [E2] {second_word} [/E2] started before [E1] {first_word} [/E1]?'
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
def print_training_progress(
        start_time, length_of_data_loader, epoch, batch_counter, total_loss):
    """
    """
    print(f'Epoch:{epoch}, '
          f'loss:{total_loss:.2f}, '
          f'Training time:{timedelta(seconds=time.time() - start_time)}, '
          f'Epoch percent: {round((batch_counter / length_of_data_loader) * 100, 2)}')
def save_model_checkpoint(
        args, model, optimizerizer, scheduler,
        length_of_data_loader, batch_counter,
        epoch, loss):
    """
    """
    PATH = Path(f"models/model_epoch_{epoch}_iter_{batch_counter}_.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizerizer_state_dict': optimizerizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss / args.save_model_every,
        'epoch percent': round((batch_counter / length_of_data_loader) * 100, 2)
    }, PATH)

    print(f'checkpoint has been saved !')
    print(f'Epoch percent: {round((batch_counter / length_of_data_loader) * 100, 2)}')
def load_model_checkpoint(path_, model, optimizer=None, scheduler=None):
    """
    """

    # load the checkpoint:
    checkpoint = torch.load(path_)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    loss = checkpoint['loss']

    epoch_percent = checkpoint['epoch percent']

    return model, optimizer, scheduler, loss, epoch_percent
def train_tre_new_questions_with_markers(
        model, args, train_dataloader, test_dataloader,
        tokenizer, num_epochs, checkpoint_path=None):
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
    :param checkpoint_path:
    :type checkpoint_path:
    :return:
    :rtype:
    """

    print('training tre with markers...')

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # loss progress counters
    total_loss_for_print = 0
    total_loss_for_save = 0

    # evaluation scores:
    eval_scores = []

    # if there is a checkpoint, load it:
    if checkpoint_path is not None:
        (model, optimizer, scheduler, _, _) = \
            load_model_checkpoint(
                checkpoint_path, model,
                optimizer, scheduler
            )

    # training mode:
    model.train()

    # start time:
    t0 = time.time()

    for epoch in range(1, num_epochs+1, 1):

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for batch_counter, instances in enumerate(train_dataloader, start=1):

            passages = instances[0]
            first_words = instances[1][0]
            second_words = instances[1][1]
            word_labels = instances[1][4]

            zip_object = zip(passages, first_words, second_words, word_labels)
            for passage, first_word, second_word, Label in zip_object:

                # ignor vague, like other papers do
                if Label.strip() == 'VAGUE':
                    continue

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
                        optimizer.zero_grad()

                        # forward pass:
                        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

                        # extract loss
                        loss = criterion(outputs, batch_labels)

                        # calculate loss
                        loss.backward()

                        # This is to help prevent the "exploding gradients" problem:
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

                        # update parameters
                        optimizer.step()

                        # Update the learning rate.
                        scheduler.step()

                        # save training loss:
                        total_loss_for_print += loss.item()
                        total_loss_for_save += loss.item()

                        batch_input_ids = []
                        batch_attention_mask = []
                        batch_labels = []

            # Print the model progress once in a while...
            if (batch_counter) % args.print_loss_every == 0:
                print_training_progress(
                    t0, len(train_dataloader),
                    epoch, batch_counter, total_loss_for_print
                )
                total_loss_for_print = 0

            # save the model once in a while, and evaluate.
            if (batch_counter) % args.save_model_every == 0:

                # save:
                if args.save_model_during_training:
                    save_model_checkpoint(
                        args, model, optimizer,
                        scheduler, len(train_dataloader),
                        batch_counter, epoch,
                        total_loss_for_save
                    )
                    total_loss_for_save = 0

                # evaluate:
                if args.eval_during_training:
                    epoch_precent = (batch_counter / len(train_dataloader)) * 100
                    tracker = results_tracker()
                    (macro, micro) = eval_tre_new_questions_with_markers(
                        model, args, test_dataloader,
                        tokenizer, tracker
                    )
                    eval_scores.append([epoch, epoch_precent, macro, micro])

        # save in the end of the epoch:
        if args.save_model_during_training:
            save_model_checkpoint(
                args, model, optimizer,
                scheduler, len(train_dataloader),
                batch_counter, epoch,
                total_loss_for_save
            )
            total_loss_for_save = 0

    return eval_scores
def eval_tre_new_questions_with_markers(
        model, args, test_dataloader,
        tokenizer, tracker, checkpoint_path=None):
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

    # if there is a checkpoint_path, then load it:
    if checkpoint_path is not None:
        (model, _, _, _, _) = \
            load_model_checkpoint(checkpoint_path, model)

    model.eval()

    print_every = 10

    for batch_counter, instances in enumerate(test_dataloader, start=1):

        passages = instances[0]
        first_words, second_words = instances[1][0], instances[1][1]
        word_labels = instances[1][4]

        zip_object = zip(passages, first_words, second_words, word_labels)
        for passage, first_word, second_word, Label in zip_object:

            # ignor vague:
            if Label.strip() == 'VAGUE':
                continue

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
            tracker.update(Label, ans1, ans2)

        if batch_counter % print_every == 0:
            macro, micro = tracker.f1_macro_and_micro()
            eval_precent = (batch_counter / len(test_dataloader)) * 100
            print(f'f1 macro: {macro}, f1 micro: {micro}, '
                  f'evaluation percent: {eval_precent:.3f}')

    # return the last f1_macro and f1_micro
    f1_macro, f1_micro = tracker.f1_macro_and_micro()

    return (f1_macro, f1_micro)
"============================================================================="
