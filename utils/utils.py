# questions for markers ([E1] {first_word} [/E1]):
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
# questions for regular markers (@ word @):


def question_1_for_regular_markers(first_word, second_word):
    """
    :param first_word:
    :type first_word:
    :param second_word:
    :type second_word:
    :return:
    :rtype:
    """
    res = f'Is it possible that @ {first_word} @ started before @ {second_word} @ ?'
    return res


def question_2_for_regular_markers(first_word, second_word):
    """
    :param first_word:
    :type first_word:
    :param second_word:
    :type second_word:
    :return:
    :rtype:
    """
    res = f'Is it possible that @ {second_word} @ started before @ {first_word} @ ?'
    return res
# questions:


def question_1(args, first_word, second_word):
    """
    :param first_word:
    :type first_word:
    :param second_word:
    :type second_word:
    :return:
    :rtype:
    """
    if args.use_E_markers:
        res = question_1_for_markers(first_word, second_word)
    else:
        res = question_1_for_regular_markers(first_word, second_word)
    return res


def question_2(args, first_word, second_word):
    """
    :param first_word:
    :type first_word:
    :param second_word:
    :type second_word:
    :return:
    :rtype:
    """
    if args.use_E_markers:
        res = question_2_for_markers(first_word, second_word)
    else:
        res = question_2_for_regular_markers(first_word, second_word)
    return res
# class that computes l1 scores (macro and micro):


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

            if ans1 == 1 and ans2 == 0:  # BEFORE
                self.TP_BEFORE += 1
                res = 'BEFORE'

            if ans1 == 0 and ans2 == 1:  # AFTER
                self.FP_AFTER += 1
                self.FN_BEFORE += 1
                res = 'AFTER'

            if ans1 == 1 and ans2 == 1:  # VAGUE
                self.TP_BEFORE += 1
                res = 'BEFORE'

            if ans1 == 0 and ans2 == 0:  # EQUAL
                self.FP_EQUAL += 1
                self.FN_BEFORE += 1
                res = 'EQUAL'

        elif label.strip() == 'AFTER':

            if ans1 == 1 and ans2 == 0:  # BEFORE
                self.FP_BEFORE += 1
                self.FN_AFTER += 1
                res = 'BEFORE'

            if ans1 == 0 and ans2 == 1:  # AFTER
                self.TP_AFTER += 1
                res = 'AFTER'

            if ans1 == 1 and ans2 == 1:  # VAGUE
                self.FP_BEFORE += 1
                self.FN_AFTER += 1
                res = 'BEFORE'

            if ans1 == 0 and ans2 == 0:  # EQUAL
                self.FP_EQUAL += 1
                self.FN_AFTER += 1
                res = 'EQUAL'

        elif (label.strip() == 'EQUAL') or (label.strip() == 'SIMULTANEOUS'):

            if ans1 == 1 and ans2 == 0:  # BEFORE
                self.FP_BEFORE += 1
                self.FN_EQUAL += 1
                res = 'BEFORE'

            if ans1 == 0 and ans2 == 1:  # AFTER
                self.FP_AFTER += 1
                self.FN_EQUAL += 1
                res = 'AFTER'

            if ans1 == 1 and ans2 == 1:  # VAGUE
                self.FP_BEFORE += 1
                self.FN_EQUAL += 1
                res = 'BEFORE'

            if ans1 == 0 and ans2 == 0:  # EQUAL
                self.TP_EQUAL += 1
                res = 'EQUAL'

        elif label.strip() == 'VAGUE':

            if ans1 == 1 and ans2 == 0:  # BEFORE
                self.TP_VAGUE += 1
                res = 'VAGUE'

            if ans1 == 0 and ans2 == 1:  # AFTER
                self.TP_VAGUE += 1
                res = 'VAGUE'

            if ans1 == 1 and ans2 == 1:  # VAGUE
                self.TP_BEFORE += 1
                res = 'VAGUE'

            if ans1 == 0 and ans2 == 0:  # EQUAL
                self.FP_EQUAL += 1
                self.FN_VAGUE += 1
                res = 'EQUAL'

        else:
            raise Exception(f'label: {label.strip()} is incorect')

        return res

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
            precision_before = self.TP_BEFORE / \
                (self.TP_BEFORE + self.FP_BEFORE)
        except (ZeroDivisionError, RuntimeWarning):
            precision_before = 0
        try:
            recall_before = self.TP_BEFORE / (self.TP_BEFORE + self.FN_BEFORE)
        except (ZeroDivisionError, RuntimeWarning):
            recall_before = 0
        try:
            f1_before = 2 * (precision_before * recall_before) / \
                (precision_before + recall_before)
        except (ZeroDivisionError, RuntimeWarning):
            f1_before = 0
        "====================================================================================="
        "AFTER"
        try:
            precision_after = self.TP_AFTER / (self.TP_AFTER + self.FP_AFTER)
        except (ZeroDivisionError, RuntimeWarning):
            precision_after = 0

        try:
            recall_after = self.TP_AFTER / (self.TP_AFTER + self.FN_AFTER)
        except (ZeroDivisionError, RuntimeWarning):
            recall_after = 0

        try:
            f1_after = 2 * (precision_after * recall_after) / \
                (precision_after + recall_after)
        except (ZeroDivisionError, RuntimeWarning):
            f1_after = 0
        "====================================================================================="
        "EQUAL"
        try:
            if self.TP_EQUAL == 0:
                precision_equal = 0
            else:
                precision_equal = self.TP_EQUAL / \
                    (self.TP_EQUAL + self.FP_EQUAL)
        except (ZeroDivisionError, RuntimeWarning):
            precision_equal = 0

        try:
            if self.TP_EQUAL == 0:
                recall_equal = 0
            else:
                recall_equal = self.TP_EQUAL / (self.TP_EQUAL + self.FN_EQUAL)
        except (ZeroDivisionError, RuntimeWarning):
            recall_equal = 0

        try:
            f1_equal = 2 * (precision_equal * recall_equal) / \
                (precision_equal + recall_equal)
        except (ZeroDivisionError, RuntimeWarning):
            f1_equal = 0
        "====================================================================================="
        "VAGUE"
        try:
            if self.TP_VAGUE == 0:
                precision_vague = 0
            else:
                precision_vague = self.TP_VAGUE / \
                    (self.TP_VAGUE + self.FP_VAGUE)
        except (ZeroDivisionError, RuntimeWarning):
            precision_vague = 0

        try:
            if self.TP_VAGUE == 0:
                recall_vague = 0
            else:
                recall_vague = self.TP_VAGUE / (self.TP_VAGUE + self.FN_VAGUE)
        except (ZeroDivisionError, RuntimeWarning):
            recall_vague = 0

        try:
            f1_vague = 2 * (precision_vague * recall_vague) / \
                (precision_vague + recall_vague)
        except (ZeroDivisionError, RuntimeWarning):
            f1_vague = 0
        "====================================================================================="
        "F1, MACRO, MICRO"

        # macro f1, just the everage:
        macro_f1 = (f1_before + f1_after + f1_equal + f1_vague) / 4

        # micro f1
        TP_sum_all_classes = self.TP_BEFORE + \
            self.TP_AFTER + self.TP_EQUAL + self.TP_VAGUE
        FP_sum_all_classes = self.FP_BEFORE + \
            self.FP_AFTER + self.FP_EQUAL + self.FP_VAGUE
        FN_sum_all_classes = self.FN_BEFORE + \
            self.FN_AFTER + self.FN_EQUAL + self.FN_VAGUE

        try:
            micro_precision = TP_sum_all_classes / \
                (TP_sum_all_classes + FP_sum_all_classes)
        except (ZeroDivisionError, RuntimeWarning):
            micro_precision = 0
        try:
            micro_recall = TP_sum_all_classes / \
                (TP_sum_all_classes + FN_sum_all_classes)
        except (ZeroDivisionError, RuntimeWarning):
            micro_recall = 0
        try:
            micro_f1 = 2 * (micro_precision * micro_recall) / \
                (micro_precision + micro_recall)
        except (ZeroDivisionError, RuntimeWarning):
            micro_f1 = 0

        return (float(f'{macro_f1:.4f}'), float(f'{micro_f1:.4f}'))

    def reset(self):

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

    def get_list_of_values(self):
        """
        this function is for dist.all_reduce in the eval phase
        :return: list of all values
        :rtype:
        """
        return [
            self.TP_BEFORE, self.TN_BEFORE, self.FP_BEFORE, self.FN_BEFORE,
            self.TP_AFTER, self.TN_AFTER, self.FP_AFTER, self.FN_AFTER,
            self.TP_EQUAL, self.TN_EQUAL, self.FP_EQUAL, self.FN_EQUAL,
            self.TP_VAGUE, self.TN_VAGUE, self.FP_VAGUE, self.FN_VAGUE
        ]

    def update_values_from_list(self, list):
        """
        this function is for updating the values after dist.all_reduce
        :param list: list (or numpy array) of reduced results
        :type list:
        :return:
        :rtype:
        """
        self.TP_BEFORE, self.TN_BEFORE, self.FP_BEFORE, self.FN_BEFORE,\
            self.TP_AFTER, self.TN_AFTER, self.FP_AFTER, self.FN_AFTER,\
            self.TP_EQUAL, self.TN_EQUAL, self.FP_EQUAL, self.FN_EQUAL,\
            self.TP_VAGUE, self.TN_VAGUE, self.FP_VAGUE, self.FN_VAGUE = list


class baseline_results_tracker:

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

    def update(self, pred, label):
        """
        :param pred: number between 0 to 3, (0-before), (1-after), (2-vague), (3-equal)
        :type pred: int
        :param label: the lable, like: 'BEFORE' etc
        :type label: str
        :return:
        :rtype:
        """

        if label.strip() == 'BEFORE':

            if pred == 0:  # BEFORE
                self.TP_BEFORE += 1
                res = 'BEFORE'

            if pred == 1:  # AFTER
                self.FP_AFTER += 1
                self.FN_BEFORE += 1
                res = 'AFTER'

            if pred == 2:  # VAGUE
                self.TP_BEFORE += 1
                res = 'BEFORE'

            if pred == 3:  # EQUAL
                self.FP_EQUAL += 1
                self.FN_BEFORE += 1
                res = 'EQUAL'

        elif label.strip() == 'AFTER':

            if pred == 0:  # BEFORE
                self.FP_BEFORE += 1
                self.FN_AFTER += 1
                res = 'BEFORE'

            if pred == 1:  # AFTER
                self.TP_AFTER += 1
                res = 'AFTER'

            if pred == 2:  # VAGUE
                self.FP_BEFORE += 1
                self.FN_AFTER += 1
                res = 'BEFORE'

            if pred == 3:  # EQUAL
                self.FP_EQUAL += 1
                self.FN_AFTER += 1
                res = 'EQUAL'

        elif (label.strip() == 'EQUAL') or (label.strip() == 'SIMULTANEOUS'):

            if pred == 0:  # BEFORE
                self.FP_BEFORE += 1
                self.FN_EQUAL += 1
                res = 'BEFORE'

            if pred == 1:  # AFTER
                self.FP_AFTER += 1
                self.FN_EQUAL += 1
                res = 'AFTER'

            if pred == 2:  # VAGUE
                self.FP_BEFORE += 1
                self.FN_EQUAL += 1
                res = 'BEFORE'

            if pred == 3:  # EQUAL
                self.TP_EQUAL += 1
                res = 'EQUAL'

        elif label.strip() == 'VAGUE':

            if pred == 0:  # BEFORE
                self.TP_VAGUE += 1
                res = 'VAGUE'

            if pred == 1:  # AFTER
                self.TP_VAGUE += 1
                res = 'VAGUE'

            if pred == 2:  # VAGUE
                self.TP_BEFORE += 1
                res = 'VAGUE'

            if pred == 3:  # EQUAL
                self.FP_EQUAL += 1
                self.FN_VAGUE += 1
                res = 'EQUAL'

        else:
            raise Exception(f'label: {label.strip()} is incorect')

        return res

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
            precision_before = self.TP_BEFORE / \
                (self.TP_BEFORE + self.FP_BEFORE)
        except (ZeroDivisionError, RuntimeWarning):
            precision_before = 0
        try:
            recall_before = self.TP_BEFORE / (self.TP_BEFORE + self.FN_BEFORE)
        except (ZeroDivisionError, RuntimeWarning):
            recall_before = 0
        try:
            f1_before = 2 * (precision_before * recall_before) / \
                (precision_before + recall_before)
        except (ZeroDivisionError, RuntimeWarning):
            f1_before = 0
        "====================================================================================="
        "AFTER"
        try:
            precision_after = self.TP_AFTER / (self.TP_AFTER + self.FP_AFTER)
        except (ZeroDivisionError, RuntimeWarning):
            precision_after = 0

        try:
            recall_after = self.TP_AFTER / (self.TP_AFTER + self.FN_AFTER)
        except (ZeroDivisionError, RuntimeWarning):
            recall_after = 0

        try:
            f1_after = 2 * (precision_after * recall_after) / \
                (precision_after + recall_after)
        except (ZeroDivisionError, RuntimeWarning):
            f1_after = 0
        "====================================================================================="
        "EQUAL"
        try:
            if self.TP_EQUAL == 0:
                precision_equal = 0
            else:
                precision_equal = self.TP_EQUAL / \
                    (self.TP_EQUAL + self.FP_EQUAL)
        except (ZeroDivisionError, RuntimeWarning):
            precision_equal = 0

        try:
            if self.TP_EQUAL == 0:
                recall_equal = 0
            else:
                recall_equal = self.TP_EQUAL / (self.TP_EQUAL + self.FN_EQUAL)
        except (ZeroDivisionError, RuntimeWarning):
            recall_equal = 0

        try:
            f1_equal = 2 * (precision_equal * recall_equal) / \
                (precision_equal + recall_equal)
        except (ZeroDivisionError, RuntimeWarning):
            f1_equal = 0
        "====================================================================================="
        "VAGUE"
        try:
            if self.TP_VAGUE == 0:
                precision_vague = 0
            else:
                precision_vague = self.TP_VAGUE / \
                    (self.TP_VAGUE + self.FP_VAGUE)
        except (ZeroDivisionError, RuntimeWarning):
            precision_vague = 0

        try:
            if self.TP_VAGUE == 0:
                recall_vague = 0
            else:
                recall_vague = self.TP_VAGUE / (self.TP_VAGUE + self.FN_VAGUE)
        except (ZeroDivisionError, RuntimeWarning):
            recall_vague = 0

        try:
            f1_vague = 2 * (precision_vague * recall_vague) / \
                (precision_vague + recall_vague)
        except (ZeroDivisionError, RuntimeWarning):
            f1_vague = 0
        "====================================================================================="
        "F1, MACRO, MICRO"

        # macro f1, just the everage:
        macro_f1 = (f1_before + f1_after + f1_equal + f1_vague) / 4

        # micro f1
        TP_sum_all_classes = self.TP_BEFORE + \
            self.TP_AFTER + self.TP_EQUAL + self.TP_VAGUE
        FP_sum_all_classes = self.FP_BEFORE + \
            self.FP_AFTER + self.FP_EQUAL + self.FP_VAGUE
        FN_sum_all_classes = self.FN_BEFORE + \
            self.FN_AFTER + self.FN_EQUAL + self.FN_VAGUE

        try:
            micro_precision = TP_sum_all_classes / \
                (TP_sum_all_classes + FP_sum_all_classes)
        except (ZeroDivisionError, RuntimeWarning):
            micro_precision = 0
        try:
            micro_recall = TP_sum_all_classes / \
                (TP_sum_all_classes + FN_sum_all_classes)
        except (ZeroDivisionError, RuntimeWarning):
            micro_recall = 0
        try:
            micro_f1 = 2 * (micro_precision * micro_recall) / \
                (micro_precision + micro_recall)
        except (ZeroDivisionError, RuntimeWarning):
            micro_f1 = 0

        return (float(f'{macro_f1:.4f}'), float(f'{micro_f1:.4f}'))

    def reset(self):

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

    def get_list_of_values(self):
        """
        this function is for dist.all_reduce in the eval phase
        :return: list of all values
        :rtype:
        """
        return [
            self.TP_BEFORE, self.TN_BEFORE, self.FP_BEFORE, self.FN_BEFORE,
            self.TP_AFTER, self.TN_AFTER, self.FP_AFTER, self.FN_AFTER,
            self.TP_EQUAL, self.TN_EQUAL, self.FP_EQUAL, self.FN_EQUAL,
            self.TP_VAGUE, self.TN_VAGUE, self.FP_VAGUE, self.FN_VAGUE
        ]

    def update_values_from_list(self, list):
        """
        this function is for updating the values after dist.all_reduce
        :param list: list (or numpy array) of reduced results
        :type list:
        :return:
        :rtype:
        """
        self.TP_BEFORE, self.TN_BEFORE, self.FP_BEFORE, self.FN_BEFORE,\
            self.TP_AFTER, self.TN_AFTER, self.FP_AFTER, self.FN_AFTER,\
            self.TP_EQUAL, self.TN_EQUAL, self.FP_EQUAL, self.FN_EQUAL,\
            self.TP_VAGUE, self.TN_VAGUE, self.FP_VAGUE, self.FN_VAGUE = list
# get the label (number) from label string:


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


def get_label_for_baseline(label):
    """
    :param label:
    :type label:
    :return:
    :rtype:
    """

    if label.strip() == 'BEFORE':
        res = [0]
    elif label.strip() == 'AFTER':
        res = [1]
    elif label.strip() == 'VAGUE':
        res = [2]
    elif label.strip() == 'EQUAL':
        res = [3]

    return res
# for multiprocess:


def all_equal(list_of_mp_values):
    """
    :param list_of_mp_values:
    :type list_of_mp_values:
    :return:
    :rtype:
    """
    return all([i.value == list_of_mp_values[0].value for i in list_of_mp_values])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
