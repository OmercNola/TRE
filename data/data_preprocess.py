from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from xml.dom import minidom
import os
import json
import torch
import itertools
from pathlib import Path, PureWindowsPath, PurePosixPath
import platform
from datetime import datetime
from collections import namedtuple
import xml
from ipdb import set_trace
import nlpaug.augmenter.char as char_aug
import nlpaug.augmenter.word as word_aug 
import nlpaug.augmenter.sentence as sentence_aug
import random
from ipdb import set_trace
"================================================================================="


def process_data(tml_folder_path, annotation_file_path):
    """
    :param tml_folder_path:
    :param annotation_file_path:
    :param tokenizer:
    :param bert_style:
    :return:
    """
    # one big file:
    annotations = read_labeled_data_file(annotation_file_path)
    texts = {}
    for file in annotations.keys():
        # file is like ABC19980114.1830.0611
        Text_elements, eiid_to_eid_map, = get_text_and_eiid_to_eid_map_from_tml_file(
            os.path.join(tml_folder_path, file + ".tml"))
        texts[file] = {"Text_elements": Text_elements,
                       "eiid_to_eid_map": eiid_to_eid_map}
    return annotations, texts


def read_labeled_data_file(filepath):
    """"
    THIS FILE IS WITH eiid AND NOT eid.
    :param filepath:
    :return:
    """

    annotations = defaultdict(list)
    with open(filepath, mode="r") as f:
        for line in f:
            # NYT20000406.0002	said	made	1	2	AFTER
            line_comp = line.split("\t")
            annotations[line_comp[0]].append(line_comp[1:])
    return annotations


def get_text_and_eiid_to_eid_map_from_tml_file(filepath):
    """
    Read specific tml file and process it,
    :param filepath:
    :param tokenizer:
    :param bert_style:
    :return:
    """
    mydoc = minidom.parse(filepath)
    elements = mydoc.getElementsByTagName("MAKEINSTANCE")
    ei_e_map = {}
    for e in elements:
        ei_e_map[e.attributes["eiid"].value] = e.attributes["eventID"].value
    elements = mydoc.getElementsByTagName("TEXT")

    return elements, ei_e_map


def new_context_with_markers_from_tokens_and_two_eids(
        args, elements_, eid1_, eid2_):
    """
    :param elements_:
    :type elements_:
    :param eid1_:
    :type eid1_:
    :param eid2_:
    :type eid2_:
    :param Shorten_text:
    :type Shorten_text:
    :return:
    :rtype:
    """

    new_context_ = ""

    for index, child in enumerate(elements_[0].childNodes):

        # e1:
        if isinstance(child, minidom.Element) and child.nodeName == "EVENT" and \
                child.attributes["eid"].value == eid1_:
            if args.use_E_markers:
                new_context_ += f'[E1] {child.firstChild.data} [/E1]'
            else:  # use @ markers
                new_context_ += f'@ {child.firstChild.data} @'

        # e2:
        elif isinstance(child, minidom.Element) and child.nodeName == "EVENT" and \
                child.attributes["eid"].value == eid2_:
            if args.use_E_markers:
                new_context_ += f'[E2] {child.firstChild.data} [/E2]'
            else:
                new_context_ += f'@ {child.firstChild.data} @'

        elif isinstance(child, minidom.Element) and (child.nodeName == "TIMEX" or child.nodeName == "TIMEX3"):
            new_context_ += child.firstChild.data

        elif isinstance(child, minidom.Element):
            new_context_ += child.firstChild.data

        else:
            new_context_ += child.data

    return new_context_


def new_short_context_with_markers_from_tokens_and_two_eids(
        args, elements_, eid1_, eid2_, data_aug):
    """
    :param elements_:
    :type elements_:
    :param eid1_:
    :type eid1_:
    :param eid2_:
    :type eid2_:
    :param Shorten_text:
    :type Shorten_text:
    :return:
    :rtype:
    """
    e2_was_found = False
    first_word = None
    new_short_context = ""
    words_that_comes_with_dot = [
        'Jan', 'Feb', 'Mar', 'Apr',
        'May', 'June', 'Aug', 'Sep',
        'Oct', 'Nov', 'Dec',
        'Prof', 'Mr', 'Inc']

    for index, child in enumerate(elements_[0].childNodes):

        # e1:
        if isinstance(child, minidom.Element) and child.nodeName == "EVENT" and \
                child.attributes["eid"].value == eid1_:
            if args.use_E_markers:
                new_short_context += f'[E1] {child.firstChild.data} [/E1]'
            else:  # use @ markers
                new_short_context += f'@ {child.firstChild.data} @'
                first_word = child.firstChild.data

        # e2:
        elif isinstance(child, minidom.Element) and child.nodeName == "EVENT" and \
                child.attributes["eid"].value == eid2_:
            if args.use_E_markers:
                new_short_context += f'[E2] {child.firstChild.data} [/E2]'
            else:
                new_short_context += f'@ {child.firstChild.data} @'
            e2_was_found = True

        elif isinstance(child, minidom.Element) and (child.nodeName == "TIMEX" or child.nodeName == "TIMEX3"):
            if e2_was_found:
                if "." in child.firstChild.data:
                    data_before_the_dot = child.firstChild.data.split(".")[0]
                    if all(word not in data_before_the_dot for word in words_that_comes_with_dot):
                        new_short_context += data_before_the_dot + '.'
                        break
            new_short_context += child.firstChild.data

        elif isinstance(child, minidom.Element):
            if e2_was_found:
                if "." in child.firstChild.data:
                    data_before_the_dot = child.firstChild.data.split(".")[0]
                    if all(word not in data_before_the_dot for word in words_that_comes_with_dot):
                        new_short_context += data_before_the_dot + '.'
                        break
            new_short_context += child.firstChild.data

        # not a minidom element:
        else:
            if e2_was_found:
                if "." in child.data:
                    data_before_the_dot = child.data.split(".")[0]
                    if all(word not in data_before_the_dot for word in words_that_comes_with_dot):
                        new_short_context += data_before_the_dot + '.'
                        break
            new_short_context += child.data
    


    #if '@ raided @ two homes' in new_short_context:
    #    print(new_short_context)
    #    set_trace()
    if first_word is None:
        return []  
    new_short_context = make_the_data_even_shorter(new_short_context, first_word)
    if data_aug:
        new_short_context = aug_data(new_short_context) 
    return [new_short_context]


def make_the_data_even_shorter(passage: str, first_word) -> str:

    words_that_comes_with_dot = [
        'Jan', 'Feb', 'Mar', 'Apr',
        'May', 'June', 'Aug', 'Sep',
        'Oct', 'Nov', 'Dec',
        'Prof', 'Mr', 'Inc']

    ## find first word:
    first_word_index = passage.find(f'@ {first_word} @')
    for idx in range(first_word_index, 0, -1):
        char = passage[idx]
        if char == '.':
            last_word_in_sentence_until_dot = passage[:idx+1].split(" ")[-1]
            if all(word not in last_word_in_sentence_until_dot\
                   for word in words_that_comes_with_dot):

                break
    
    if idx < 5:
        return passage
    
    new_passage = passage[(idx +1):] 
    while (new_passage[0] == " ") or (new_passage[0] == ','):
        new_passage = new_passage[1:]

    return new_passage



def aug_data(passage):

    bag_of_augmentations = ['char', 'sentence']
    aug_method = random.choice(bag_of_augmentations)
    split_passage = passage.split('@')

    first_part_of_sentence = split_passage[0]
    second_part_of_sentence = split_passage[2]
    third_part_of_sentence = split_passage[4]

    assert len(split_passage)==5, 'number of "@" in the passage should be 4'
	
    if aug_method == 'char':
        
        aug = char_aug.OcrAug()

        first_part_aug = aug.augment(first_part_of_sentence, n=1)
        second_part_aug = aug.augment(second_part_of_sentence, n=1)
        third_part_aug = aug.augment(third_part_of_sentence, n=1)

        new_aug_sentence = "".join(first_part_aug +
                                   [' @' + split_passage[1] + '@ '] +
                                   second_part_aug +
                                   [' @' + split_passage[3] + '@ '] +
                                   third_part_aug)  
        
    elif aug_method == 'word':
        aug = word_aug.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="insert")
        first_part_aug = aug.augment(first_part_of_sentence, n=1)
        second_part_aug = aug.augment(second_part_of_sentence, n=1)
        third_part_aug = aug.augment(third_part_of_sentence, n=1)
        new_aug_sentence = "".join(first_part_aug +
                                   [' @' + split_passage[1]+'@ '] +
                                   second_part_aug +
                                   [' @' + split_passage[3]+'@ '] +
                                   third_part_aug)  
        
    elif aug_method == 'sentence':
        new_aug_sentence = passage

    return new_aug_sentence


def final_data_process_for_markers(args, folder_path, labeled_data_path, data_aug=False):
    """
    :param folder_path:
    :type folder_path:
    :param labeled_data_path:
    :type labeled_data_path:
    :return:
    :rtype:
    """
    data = []
    max_passage_length = 0
    TimeBank_labeled_data, TimeBank_Dict_with_Text_elements_and_ei_map = \
        process_data(folder_path, labeled_data_path)

    for key in list(TimeBank_labeled_data.keys()):

        Map = TimeBank_Dict_with_Text_elements_and_ei_map[key]['eiid_to_eid_map']
        text_elements = TimeBank_Dict_with_Text_elements_and_ei_map[key]['Text_elements']

        for instance in TimeBank_labeled_data[key]:

            # instance is like: ['predicted', 'tried', '415', '417',
            # 'BEFORE\n']
            if len(instance) != 5:
               continue 
            eiid1 = instance[2]
            eiid2 = instance[3]

            try:
                eid1 = Map[f'ei{eiid1}']
                eid2 = Map[f'ei{eiid2}']

                if args.short_passage:
                    # here we get passage with markers and cut it just after
                    # the first "." after [E2]:
                    passages = new_short_context_with_markers_from_tokens_and_two_eids(
                        args, text_elements, eid1, eid2, data_aug)
                    max_passage_length = max(max_passage_length, len(passages[0].split(" ")))
                else:
                    # here we get all passage with markers:
                    passage = new_context_with_markers_from_tokens_and_two_eids(
                        args, text_elements, eid1, eid2)

                first_word = instance[0]
                second_word = instance[1]
                relation = instance[4]
                for passage in passages:

                    data.append([passage, [first_word, second_word, relation]])

            except BaseException:
                pass
    print(f'max_passage_length:{max_passage_length}')
    return data


"================================================================================="


def get_text_and_labeled_data_from_tml_file_tcr(filepath):
    """
    Read specific tml file and process it,
    :param filepath:
    :param tokenizer:
    :param bert_style:
    :return:
    """
    mydoc = minidom.parse(filepath)
    "=========================================================================="
    elements = mydoc.getElementsByTagName("MAKEINSTANCE")
    ei_e_map = {}
    for e in elements:
        ei_e_map[e.attributes["eiid"].value] = e.attributes["eventID"].value
    "=========================================================================="
    text_elements = mydoc.getElementsByTagName("TEXT")
    "=========================================================================="
    labeled_data = []
    tlink_elements = mydoc.getElementsByTagName("TLINK")

    for e in tlink_elements:

        if ('relatedToEventInstance' not in list(e.attributes.keys())) or \
                ('eventInstanceID' not in list(e.attributes.keys())):
            continue

        eventID_1 = ei_e_map[e.attributes["eventInstanceID"].value]
        eventID_2 = ei_e_map[e.attributes["relatedToEventInstance"].value]
        relation = e.attributes['relType'].value

        for index, child in enumerate(text_elements[0].childNodes):
            if isinstance(child, minidom.Element) and child.nodeName == "EVENT" and \
                    child.attributes["eid"].value == eventID_1:
                word_1 = child.firstChild.data
            if isinstance(child, minidom.Element) and child.nodeName == "EVENT" and \
                    child.attributes["eid"].value == eventID_2:
                word_2 = child.firstChild.data

        labeled_data.append(
            (
                (eventID_1, word_1),
                (eventID_2, word_2),
                relation,
                text_elements
            )
        )
    "=========================================================================="
    return labeled_data


def process_TCR_data(args, tml_folder_path, use_augmentation=True):
    """
    :param tml_folder_path:
    :type tml_folder_path:
    :return:
    :rtype:
    """

    # our data, will be like: [passage, instance]
    # passage is the result from new_context_with_markers_from_tokens_and_two_eids func
    # instance is like: [first_word, second_word, relation]
    data = []
    File = namedtuple('File', 'name path size modified_date')
    tml_files = []
    for item in tml_folder_path.glob('**/*'):
        if item.suffix == '.tml':
            name = item.name
            path = Path.resolve(item)
            size = item.stat().st_size
            modified = datetime.fromtimestamp(item.stat().st_mtime)
            tml_files.append(File(name, path, size, modified))

    for file in tml_files:

        labeled_data = get_text_and_labeled_data_from_tml_file_tcr(
            os.path.join(file.path))

        for event1, event2, relation, text_elements in labeled_data:

            eid1, first_word = event1[0], event1[1]
            eid2, second_word = event2[0], event2[1]

            if args.short_passage:
                # here we get passage with markers and cut it just after the
                # first "." after [E2]:
                passage = new_short_context_with_markers_from_tokens_and_two_eids(
                    args, text_elements, eid1, eid2, use_augmentation)
            else:
                # here we get all passage with markers:
                passage = new_context_with_markers_from_tokens_and_two_eids(
                    args, text_elements, eid1, eid2
                )

            data.append([passage, [first_word, second_word, relation]])

    return data


