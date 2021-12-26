def question_answer(model_, input_ids, tokenizer):
    model_.eval()
    # tokenize question and text as a pair
    # input_ids = tokenizer.encode(question, text)

    # string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segment IDs
    # first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    # number of tokens in segment A (question)
    num_seg_a = sep_idx + 1
    # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a

    # list of 0s and 1s for segment embeddings
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    # model output using input_ids and segment_ids
    output = model_(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)

    # print(answer_start)
    # print(answer_end)
    # raise Exception

    answer = tokens[answer_start]
    # for i in range(answer_start + 1, answer_start + 2):
    #     if tokens[i][0:2] == "##":
    #         answer += tokens[i][2:]
    #     else:
    #         answer += " " + tokens[i]

    # if answer_end >= answer_start:
    #     answer = tokens[answer_start]
    #     for i in range(answer_start + 1, answer_end + 1):
    #         if tokens[i][0:2] == "##":
    #             answer += tokens[i][2:]
    #         else:
    #             answer += " " + tokens[i]

    return answer
def train(Tokenizer, data_to_train_with, dict_with_text_elements_and_ei_map, bert_style=True, num_epochs=1):

    model.train()
    optim = AdamW(model.parameters(), lr=0.00001)
    print_every = 100
    how_many_instances_did_we_missed = 0

    keys = list(data_to_train_with.keys())

    for e in range(num_epochs):

        random.shuffle(keys)
        LOSS = 0
        big_counter = 0

        for files_counter, file_ in enumerate(keys):

            elements = dict_with_text_elements_and_ei_map[file_]['Text_elements']
            map = dict_with_text_elements_and_ei_map[file_]['eiid_to_eid_map']

            for instances_counter, (first_w, second_w, eiid1, eiid2, label) in enumerate(data_to_train_with[file_]):

                big_counter += 1 # count steps in epoch
                optim.zero_grad()

                try:
                    eid1 = map[f'ei{eiid1}']
                    eid2 = map[f'ei{eiid2}']
                except Exception as e:
                    print(e)
                    print(file_)
                    continue

                question = give_me_question_from_two_words(first_w, second_w)
                new_reduced_context = give_me_new_context_from_tokens_and_two_eids(elements, eid1, eid2)

                # tokenize question and text as a pair
                encodings = Tokenizer(question, new_reduced_context)

                if len(encodings['input_ids']) > 512:
                    how_many_instances_did_we_missed += 1
                    continue

                # string version of tokenized ids
                tokens = Tokenizer.convert_ids_to_tokens(encodings['input_ids'])

                if label.strip() == 'BEFORE':
                    label_word = first_w
                elif label.strip() == 'AFTER':
                    label_word = second_w
                elif label.strip() == 'EQUAL':
                    label_word = '[CLS]'
                elif label.strip() == 'VAGUE':
                    label_word = '[SEP]'

                pos_of_label_at_the_tokens_list = give_me_pos_in_text(tokens, label_word)
                # print(f'pos_of_label_at_the_tokens_list:{pos_of_label_at_the_tokens_list}')

                # # segment IDs
                # # first occurrence of [SEP] token
                # sep_idx = input_ids.index(Tokenizer.sep_token_id)
                # # number of tokens in segment A (question)
                # num_seg_a = sep_idx + 1
                # # number of tokens in segment B (text)
                # num_seg_b = len(input_ids) - num_seg_a
                # # list of 0s and 1s for segment embeddings
                # segment_ids = [0] * num_seg_a + [1] * num_seg_b
                # assert len(segment_ids) == len(input_ids)

                # model output using input_ids and segment_ids

                input_ids = torch.tensor([encodings['input_ids']], requires_grad=False).to(device)
                start_pos = torch.tensor([pos_of_label_at_the_tokens_list], requires_grad=False).to(device)
                end_pos = torch.tensor([pos_of_label_at_the_tokens_list], requires_grad=False).to(device)
                segment_ids = torch.tensor([encodings['token_type_ids']], requires_grad=False).to(device)

                outputs = model(input_ids=input_ids,
                                start_positions=start_pos,
                                end_positions=end_pos,
                                token_type_ids=segment_ids)

                # extract loss
                loss = outputs[0]
                LOSS += loss.item()

                if big_counter % print_every == 0:
                    print(f'LOSS:{LOSS}')
                    LOSS = 0
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters


                optim.step()

                # reconstructing the answer
                answer_start = torch.argmax(outputs.start_logits)
                answer_end = torch.argmax(outputs.end_logits)

                # print(f'answer_start:{answer_start}, answer_end:{answer_end}')
                try:
                    ans = tokens[answer_start:answer_end + 1]
                except:
                    ans = "can't answer"
                # print(f'ans:{ans}')

    print(f'how_many_instances_did_we_missed due to len > 512:{how_many_instances_did_we_missed}')
def eval(Tokenizer, data_to_eval_with, dict_with_text_elements_and_ei_map_, bert_style=True):

    model.eval()
    right, wrong = 0, 0
    # F1 score = 2 * (precision * recall) / (precision + recall)
    # precision = TP / TP + FP
    # recall = TP / TP + FN

    TP, FP, FN = 0, 0, 0

    for files_counter, file_ in enumerate(data_to_eval_with.keys()):

        map = dict_with_text_elements_and_ei_map_[file_]['eiid_to_eid_map']
        elements = dict_with_text_elements_and_ei_map_[file_]['Text_elements']

        for instance_counter, (first_w, second_w, eiid1, eiid2, label) in enumerate(data_to_eval_with[file_]):

            eid1 = map[f'ei{eiid1}']
            eid2 = map[f'ei{eiid2}']
            question = give_me_question_from_two_words(first_w, second_w)
            new_reduced_context = give_me_reduced_context_from_tokens_and_two_eids(elements, eid1, eid2)

            if label.strip() == 'EQUAL' or label.strip() == 'VAGUE':
                continue

            # tokenize question and text as a pair
            input_ids = Tokenizer.encode(question, new_reduced_context)

            if len(input_ids) > 512:
                continue

            # string version of tokenized ids
            tokens = Tokenizer.convert_ids_to_tokens(input_ids)

            # segment IDs
            # first occurrence of [SEP] token
            sep_idx = input_ids.index(Tokenizer.sep_token_id)
            # number of tokens in segment A (question)
            num_seg_a = sep_idx + 1
            # number of tokens in segment B (text)
            num_seg_b = len(input_ids) - num_seg_a

            # list of 0s and 1s for segment embeddings
            segment_ids = [0] * num_seg_a + [1] * num_seg_b
            assert len(segment_ids) == len(input_ids)

            # model output using input_ids and segment_ids
            input_ids = torch.tensor([input_ids]).to(device)
            segment_ids = torch.tensor([segment_ids]).to(device)

            with torch.no_grad():
                output = model(input_ids=input_ids, token_type_ids=segment_ids)

            # reconstructing the answer
            answer_start = torch.argmax(output.start_logits)
            answer_end = torch.argmax(output.end_logits)
            try:
                ans = tokens[answer_start:answer_end + 1]
            except:
                ans = ["can't answer"]

            print(ans)

            if (len(ans) > 3) or (len(ans) == 0):
                wrong += 1

            else:

                ans_len = len(ans)
                found_right_ans = False

                for i in range(ans_len):
                    ans_i = ans[i]
                    if label.strip() == 'AFTER':
                        if ans_i == second_w:
                            TP += 1
                            found_right_ans = True
                            break
                        if ans_i == first_w:
                            FN += 1

                    elif label.strip() == 'BEFORE':
                        if ans_i == first_w:
                            TP += 1
                            found_right_ans = True
                            break
                        if ans_i == second_w:
                            FN += 1

                if not found_right_ans:
                    FP += 1

    print(f'right: {TP}, wrong: {FP}, FN: {FN}')
    precision = TP / (TP + FP)
    print(f'precision: {precision}')
    recall = TP / (TP + FN)
    print(f'recall: {recall}')

    F1_score = 2 * (precision * recall) / (precision + recall)
    print(f'F1_score: {F1_score}')

def give_me_reduced_context_from_tokens_and_two_eids(elements_, eid1_, eid2_):
    """
    :param elements_:
    :param eid1_:
    :param eid2_:
    :return:
    """
    new_context_ = ""
    how_many_events_here = 0
    search_for_dot_to_end_the_passage = False
    end_for_loop = False

    for index, child in enumerate(elements_[0].childNodes):

        if end_for_loop:
            break

        if type(child) is minidom.Element and child.nodeName == "EVENT" and (child.attributes["eid"].value == eid1_):
            how_many_events_here += 1
            if how_many_events_here == 1:
                new_context_ += f'<{child.firstChild.data}>'
            elif how_many_events_here == 2:
                new_context_ += f'<{child.firstChild.data}>'
                search_for_dot_to_end_the_passage = True

        elif type(child) is minidom.Element and child.nodeName == "EVENT" and (child.attributes["eid"].value == eid2_):
            how_many_events_here += 1
            if how_many_events_here == 1:
                new_context_ += f'<{child.firstChild.data}>'
            elif how_many_events_here == 2:
                new_context_ += f'<{child.firstChild.data}>'
                search_for_dot_to_end_the_passage = True

        elif type(child) is minidom.Element and (child.nodeName == "TIMEX" or child.nodeName == "TIMEX3"):
            new_context_ += child.firstChild.data

        elif type(child) is minidom.Element:
            if search_for_dot_to_end_the_passage:
                for char in child.firstChild.data:
                    if char == ".":
                        new_context_ += child.firstChild.data.split(".")[0]
                        new_context_ += "."
                        end_for_loop = True
                if not end_for_loop:
                    new_context_ += child.firstChild.data
            else:
                new_context_ += child.firstChild.data

        else:
            if search_for_dot_to_end_the_passage:
                for char in child.data:
                    if char == ".":
                        new_context_ += child.data.split(".")[0]
                        new_context_ += "."
                        end_for_loop = True
                if not end_for_loop:
                    new_context_ += child.data
            else:
                new_context_ += child.data

    return new_context_
def give_me_pos_in_text(list_of_tokens, word_to_search_for_pos):
    """
    :param list_of_tokens:
    :return:
    """
    res = 0
    if word_to_search_for_pos == '[CLS]':
        return 0
    for index, word in enumerate(list_of_tokens):
        if word == '[SEP]':
            break
    for index_1, word in enumerate(list_of_tokens[index:]):
        if word == word_to_search_for_pos:
            res = index + index_1

    return res

