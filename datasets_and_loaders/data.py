import random
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from xml.dom import minidom
import os
import json
import torch
import itertools
import random
from pathlib import Path, PureWindowsPath, PurePosixPath
import platform
from datetime import datetime
from collections import namedtuple
import xml
from torch.utils.data import Dataset
File = namedtuple('File', 'name path size modified_date')
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
def new_context_with_markers_from_tokens_and_two_eids(elements_, eid1_, eid2_):
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
def new_short_context_with_markers_from_tokens_and_two_eids(elements_, eid1_, eid2_):
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
    new_short_context = ""

    for index, child in enumerate(elements_[0].childNodes):

        # e1:
        if type(child) is minidom.Element and child.nodeName == "EVENT" and \
                child.attributes["eid"].value == eid1_:
            new_short_context += f'@ {child.firstChild.data} @'

        # e2:
        elif type(child) is minidom.Element and child.nodeName == "EVENT" and \
                child.attributes["eid"].value == eid2_:
            new_short_context += f'@ {child.firstChild.data} @'
            e2_was_found = True


        elif type(child) is minidom.Element and (child.nodeName == "TIMEX" or child.nodeName == "TIMEX3"):
            if e2_was_found:
                if "." in child.firstChild.data:
                    new_short_context += child.firstChild.data.split(".")[0] + "."
                    break
            new_short_context += child.firstChild.data

        elif type(child) is minidom.Element:
            if e2_was_found:
                if "." in child.firstChild.data:
                    new_short_context += child.firstChild.data.split(".")[0] + "."
                    break
            new_short_context += child.firstChild.data

        # not a minidom element:
        else:
            if e2_was_found:
                if "." in child.data:
                    new_short_context += child.data.split(".")[0] + "."
                    break
            new_short_context += child.data

    return new_short_context
def final_data_process_for_markers(folder_path, labeled_data_path):
    """
    :param folder_path:
    :type folder_path:
    :param labeled_data_path:
    :type labeled_data_path:
    :return:
    :rtype:
    """
    data = []

    TimeBank_labeled_data, TimeBank_Dict_with_Text_elements_and_ei_map = \
        process_data(folder_path, labeled_data_path)

    for key in list(TimeBank_labeled_data.keys()):

        Map = TimeBank_Dict_with_Text_elements_and_ei_map[key]['eiid_to_eid_map']
        text_elements = TimeBank_Dict_with_Text_elements_and_ei_map[key]['Text_elements']

        for instance in TimeBank_labeled_data[key]:

            # instance is like: ['predicted', 'tried', '415', '417', 'BEFORE\n']

            eiid1 = instance[2]
            eiid2 = instance[3]

            try:
                eid1 = Map[f'ei{eiid1}']
                eid2 = Map[f'ei{eiid2}']

                # here we get all passage with markers:
                # passage = new_context_with_markers_from_tokens_and_two_eids(text_elements, eid1, eid2)

                # here we get passage with markers and cut it just after the first "." after [E2]:
                passage = new_short_context_with_markers_from_tokens_and_two_eids(text_elements, eid1, eid2)

                first_word = instance[0]
                second_word = instance[1]
                relation = instance[4]
                data.append([passage, [first_word, second_word, relation]])

            except:
                pass
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
            if type(child) is minidom.Element and child.nodeName == "EVENT" and \
                    child.attributes["eid"].value == eventID_1:
                word_1 = child.firstChild.data
            if type(child) is minidom.Element and child.nodeName == "EVENT" and \
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
def process_TCR_data(tml_folder_path):
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

    tml_files = []
    for item in tml_folder_path.glob('**/*'):
        if item.suffix == '.tml':
            name = item.name
            path = Path.resolve(item)
            size = item.stat().st_size
            modified = datetime.fromtimestamp(item.stat().st_mtime)
            tml_files.append(File(name, path, size, modified))


    for file in tml_files:

        labeled_data = get_text_and_labeled_data_from_tml_file_tcr(os.path.join(file.path))

        for event1, event2, relation, text_elements in labeled_data:

            eid1, first_word = event1[0], event1[1]
            eid2, second_word = event2[0], event2[1]

            # here we get all passage with markers:
            # passage = new_context_with_markers_from_tokens_and_two_eids(text_elements, eid1, eid2)

            # here we get passage with markers and cut it just after the first "." after [E2]:
            passage = new_short_context_with_markers_from_tokens_and_two_eids(text_elements, eid1, eid2)

            data.append([passage, [first_word, second_word, relation]])

    return data
"================================================================================="
class TRE_train_dataset(Dataset):

    def __init__(self, ):
        super().__init__()

        "=============================================================="
        """TimeBank"""
        TimeBank_folder = Path('./data/TBAQ-cleaned/TimeBank/')
        TimeBank_labeled_data = Path('./data/timebank.txt')
        TimeBank_data_with_markers = final_data_process_for_markers(
            TimeBank_folder, TimeBank_labeled_data
        )
        "=============================================================="
        """Aquaint"""
        Aq_folder = Path('./data/TBAQ-cleaned/AQUAINT/')
        Aq_labeled_data = Path('./data/aquaint.txt')
        Aq_data_with_markers = final_data_process_for_markers(
            Aq_folder, Aq_labeled_data
        )
        "=============================================================="
        """Aquaint and Timebank with markers (train data)"""
        self.TRE_training_data_with_markers = \
            Aq_data_with_markers + TimeBank_data_with_markers
        "=============================================================="

    def __len__(self):
        return len(self.TRE_training_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_training_data_with_markers[idx]
        return res
class TRE_val_dataset(Dataset):

    def __init__(self, ):
        super().__init__()

        "=============================================================="
        """TCR (val data)"""
        TCR_folder = Path('./data/TBAQ-cleaned/TemporalPart/')
        self.TRE_validation_data_with_markers = process_TCR_data(TCR_folder)
        "=============================================================="

    def __len__(self):
        return len(self.TRE_validation_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_validation_data_with_markers[idx]
        return res
class TRE_test_dataset(Dataset):

    def __init__(self, ):
        super().__init__()

        "=============================================================="
        """Platinum (test data)"""
        Platinum_folder = Path('./data/TBAQ-cleaned/platinum/')
        Platinum_labeled_data = Path('./data/platinum.txt')
        self.TRE_test_data_with_markers = final_data_process_for_markers(
            Platinum_folder, Platinum_labeled_data
        )
        "=============================================================="

    def __len__(self):
        return len(self.TRE_test_data_with_markers)

    def __getitem__(self, idx):
        res = self.TRE_test_data_with_markers[idx]
        return res

