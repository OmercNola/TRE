from logger import (train_log, save_model_checkpoint,
                    load_model_checkpoint, print_training_progress)
from utils import (question_1_for_markers, question_2_for_markers,
                   get_label, results_tracker)
from torch import nn
import torch
import time
from datetime import timedelta
import datetime as datetime
import random
from pathlib import Path
import wandb
"============================================================================="
# evaluation:
def eval_tre_new_questions_with_markers(
        model, args, test_dataloader,
        tokenizer, tracker, checkpoint_path=None,
        batches_overall=None):
    """
    :param model:
    :type model:
    :param args:
    :type args:
    :param test_dataloader:
    :type test_dataloader:
    :param tokenizer:
    :type tokenizer:
    :param tracker:
    :type tracker:
    :param checkpoint_path:
    :type checkpoint_path:
    :param batches_overall:
    :type batches_overall:
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

            # get f1 macro and f1 micro results:
            macro, micro = tracker.f1_macro_and_micro()

            # save then to wandb:
            wandb.log({"batches_overall": batches_overall,
                       "f1 macro": macro, "f1 micro": micro})

            eval_precent = (batch_counter / len(test_dataloader)) * 100
            print(f'f1 macro: {macro}, f1 micro: {micro}, '
                  f'evaluation percent: {eval_precent:.3f}')


    # at the end of the evaluation:
    macro, micro = tracker.f1_macro_and_micro()

    # save then to wandb:
    wandb.log({"batches_overall": batches_overall,
               "f1 macro": macro, "f1 micro": micro})

    eval_precent = (batch_counter / len(test_dataloader)) * 100
    print(f'f1 macro: {macro}, f1 micro: {micro}, '
          f'evaluation percent: {eval_precent:.3f}')
"============================================================================="
