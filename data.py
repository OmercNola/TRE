import random
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from xml.dom import minidom
import os
from transformers import AutoTokenizer, AutoModel, BertTokenizer, RobertaTokenizer, AdamW
import json
import torch
import itertools
import random
from pathlib import Path, PureWindowsPath, PurePosixPath
import platform
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
        Text_elements, eiid_to_eid_map,  = \
            get_text_and_eiid_to_eid_map_from_tml_file(os.path.join(tml_folder_path, file + ".tml"))
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
def new_context_from_tokens_and_two_eids(elements_, eid1_, eid2_):
    """
    :param elements_:
    :param eid1_:
    :param eid2_:
    :return:
    """
    new_context_ = ""

    for index, child in enumerate(elements_[0].childNodes):
        if type(child) is minidom.Element and child.nodeName == "EVENT" and \
                ((child.attributes["eid"].value == eid1_) or (child.attributes["eid"].value == eid2_)):
            new_context_ += f'<{child.firstChild.data}>'
        elif type(child) is minidom.Element and (child.nodeName == "TIMEX" or child.nodeName == "TIMEX3"):
            new_context_ += child.firstChild.data
        elif type(child) is minidom.Element:
            new_context_ += child.firstChild.data
        else:
            new_context_ += child.data

    return new_context_
def final_data_process(folder_path, labeled_data_path):
    data_ = []
    TimeBank_labeled_data, TimeBank_Dict_with_Text_elements_and_ei_map = \
        process_data(folder_path, labeled_data_path)
    for key in list(TimeBank_labeled_data.keys()):
        Map = TimeBank_Dict_with_Text_elements_and_ei_map[key]['eiid_to_eid_map']
        text_elements = TimeBank_Dict_with_Text_elements_and_ei_map[key]['Text_elements']
        for instance in TimeBank_labeled_data[key]:
            eiid1 = instance[2]
            eiid2 = instance[3]
            label = instance[4]
            # if label.strip() != 'BEFORE' and label.strip() != 'AFTER':
            #     continue
            try:
                eid1 = Map[f'ei{eiid1}']
                eid2 = Map[f'ei{eiid2}']
                passage = new_context_from_tokens_and_two_eids(text_elements, eid1, eid2)
                data_.append([passage, instance])
            except:
                pass
    return data_
# prepare the data with markers:
def new_context_with_markers_from_tokens_and_two_eids(elements_, eid1_, eid2_):
    """
    :param elements_:
    :param eid1_:
    :param eid2_:
    :return:
    """
    new_context_ = ""

    for index, child in enumerate(elements_[0].childNodes):
        if type(child) is minidom.Element and child.nodeName == "EVENT" and \
                child.attributes["eid"].value == eid1_:
            new_context_ += f'[E1] {child.firstChild.data} [/E1]'
        elif type(child) is minidom.Element and child.nodeName == "EVENT" and \
                child.attributes["eid"].value == eid2_:
            new_context_ += f'[E2] {child.firstChild.data} [/E2]'
        elif type(child) is minidom.Element and (child.nodeName == "TIMEX" or child.nodeName == "TIMEX3"):
            new_context_ += child.firstChild.data
        elif type(child) is minidom.Element:
            new_context_ += child.firstChild.data
        else:
            new_context_ += child.data

    return new_context_
def final_data_process_for_markers(folder_path, labeled_data_path):
    data_ = []
    TimeBank_labeled_data, TimeBank_Dict_with_Text_elements_and_ei_map = \
        process_data(folder_path, labeled_data_path)
    for key in list(TimeBank_labeled_data.keys()):
        Map = TimeBank_Dict_with_Text_elements_and_ei_map[key]['eiid_to_eid_map']
        text_elements = TimeBank_Dict_with_Text_elements_and_ei_map[key]['Text_elements']
        for instance in TimeBank_labeled_data[key]:
            eiid1 = instance[2]
            eiid2 = instance[3]

            try:
                eid1 = Map[f'ei{eiid1}']
                eid2 = Map[f'ei{eiid2}']
                passage = new_context_with_markers_from_tokens_and_two_eids(text_elements, eid1, eid2)
                data_.append([passage, instance])
            except:
                pass
    return data_
"================================================================================="
"""TimeBank"""
TimeBank_folder = Path('data/TBAQ-cleaned/TimeBank/')
TimeBank_labeled_data = Path('data/timebank.txt')
# without markers:
TimeBank_data = final_data_process(TimeBank_folder, TimeBank_labeled_data)
# with markers:
TimeBank_data_with_markers = final_data_process_for_markers(
    TimeBank_folder, TimeBank_labeled_data
)
"================================================================================="
"""Aquaint"""
Aq_folder = Path('data/TBAQ-cleaned/AQUAINT/')
Aq_labeled_data = Path('data/aquaint.txt')
# without markers:
Aq_data = final_data_process(Aq_folder, Aq_labeled_data)
# with markers:
Aq_data_with_markers = final_data_process_for_markers(Aq_folder, Aq_labeled_data)
"================================================================================="
"""Aquaint and Timebank with markers (val and train data)"""
Aq_and_Timebank_with_markers = Aq_data_with_markers + TimeBank_data_with_markers
print(f'len of Aq_and_Timebank_with_markers: {len(Aq_and_Timebank_with_markers)}')
TRE_validation_data_with_markers = random.sample(
    Aq_and_Timebank_with_markers,
    int(len(Aq_and_Timebank_with_markers) * 0.1)
)
print(f'len of validation data with markers: {len(TRE_validation_data_with_markers)}')
TRE_training_data_with_markers = [
    i for i in Aq_and_Timebank_with_markers
    if i not in TRE_validation_data_with_markers
]
print(f'len of train data with markers: {len(TRE_training_data_with_markers)}')
"================================================================================="
"""Platinum, (test data)"""
Platinum_folder = Path('data/TBAQ-cleaned/platinum/')
Platinum_labeled_data = Path('data/platinum.txt')
# without markers:
TRE_test_data = final_data_process(Platinum_folder, Platinum_labeled_data)
# with markers:
TRE_test_data_with_markers = final_data_process_for_markers(
    Platinum_folder, Platinum_labeled_data
)
print(f'len of test data with markers: {len(TRE_test_data_with_markers)}')
"================================================================================="