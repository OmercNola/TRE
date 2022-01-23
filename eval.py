from logger import (train_log, save_model_checkpoint,
                    load_model_checkpoint, print_training_progress)
from utils import (question_1_for_markers, question_2_for_markers,
                   question_1_for_regular_markers, question_2_for_regular_markers,
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

    print("evaluating...")

    # if there is a checkpoint_path, then load it:
    # we need just the model for evaluation
    if checkpoint_path is not None:
        (model, _, _, _, _, _) = \
            load_model_checkpoint(checkpoint_path, model)

    # evaluation mode:
    model.eval()

    # create wandb table for traking the rsults:
    table = wandb.Table(
        columns=[
            'passage', 'passage length',
            'word_1', 'word_2',
            'ans_1', 'ans_2',
            'pred_label', 'real_label',
            'correct answer'
        ]
    )

    for batch_counter, instances in enumerate(test_dataloader, start=1):

        passages = instances[0]
        first_words, second_words = instances[1][0], instances[1][1]
        word_labels = instances[1][2]

        zip_object = zip(passages, first_words, second_words, word_labels)
        for passage, first_word, second_word, Label in zip_object:

            # ignor vague, like other papers do:
            if Label.strip() == 'VAGUE':
                continue

            # get the questions:
            question_1 = question_1_for_regular_markers(
                first_word, second_word) + tokenizer.sep_token
            question_2 = question_2_for_regular_markers(
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

                # ensure no gradients for eval:
                with torch.no_grad():

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    # our prediction:
                    pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

                    # move to cpu and numpy:
                    pred = pred.clone().detach().cpu().numpy()[0]

                    # results:
                    results.append([question_name, pred])

            # now we 2 questions ready, we update results tracker:
            ans1, ans2 = results[0][1], results[1][1]
            pred_label = tracker.update(Label, ans1, ans2)

            # data for logging in wandb:
            passage_length = len(passage)

            if Label.strip() == 'SIMULTANEOUS':
                correct_answer = pred_label == 'EQUAL'
            else:
                correct_answer = pred_label == Label.strip()

            real_label = Label.strip()

            # add data to the wandb table:
            table.add_data(
                passage, passage_length, first_word, second_word,
                ans1, ans2, pred_label, real_label, correct_answer
            )

        if batch_counter % args.print_eval_every == 0:

            # get f1 macro and f1 micro results:
            macro, micro = tracker.f1_macro_and_micro()

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

    if args.save_table_of_results_after_eval:
        wandb.log({f'results table {wandb.run.name}': table})

    # finish the session just when eval:
    if args.eval:
        wandb.finish()
"============================================================================="
