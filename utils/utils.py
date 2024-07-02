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
#questions:
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

class BaseTracker:
    def __init__(self):
        self.labels = ['before', 'after', 'equal', 'vague']
        self.metrics = ['tp', 'tn', 'fp', 'fn']
        self.results = {label: {metric: 0 for metric in self.metrics}
                        for label in self.labels}

    def f1_macro_and_micro(self):
        """
        Calculate the macro and micro F1 scores.

        :return: Tuple of macro F1 and micro F1 scores
        """
        macro_f1 = sum(self._calculate_f1(label) for label in self.labels) / len(self.labels)

        tp_sum = sum(self.results[label]['tp'] for label in self.labels)
        fp_sum = sum(self.results[label]['fp'] for label in self.labels)
        fn_sum = sum(self.results[label]['fn'] for label in self.labels)

        micro_precision = self._safe_divide(tp_sum, tp_sum + fp_sum)
        micro_recall = self._safe_divide(tp_sum, tp_sum + fn_sum)
        micro_f1 = self._calculate_f1_score(micro_precision, micro_recall)

        return float(f'{macro_f1:.4f}'), float(f'{micro_f1:.4f}')

    def _calculate_f1(self, label):
        """
        Calculate the F1 score for a specific label.

        :param label: The label to calculate F1 score for
        :return: The F1 score
        """
        precision = self._safe_divide(self.results[label]['tp'],
                                      self.results[label]['tp'] +
                                      self.results[label]['fp'])
        recall = self._safe_divide(self.results[label]['tp'],
                                   self.results[label]['tp'] +
                                   self.results[label]['fn'])
        return self._calculate_f1_score(precision, recall)

    @staticmethod
    def _calculate_f1_score(precision, recall):
        """
        Calculate the F1 score from precision and recall.

        :param precision: The precision value
        :param recall: The recall value
        :return: The F1 score
        """
        return BaseTracker._safe_divide(2 * precision * recall,
                                        precision + recall)

    @staticmethod
    def _safe_divide(numerator, denominator):
        """
        Safely divide two numbers, returning 0 if denominator is 0.

        :param numerator: The numerator
        :param denominator: The denominator
        :return: The result of the division or 0 if denominator is 0
        """
        return numerator / denominator if denominator != 0 else 0

    def reset(self):
        """Reset all results to zero."""
        for label in self.results:
            for metric in self.results[label]:
                self.results[label][metric] = 0

    def get_list_of_values(self):
        """
        Get a list of all metric values.

        :return: List of all metric values
        """
        return [self.results[label][metric] for label in self.results
                for metric in self.metrics]

    def update_values_from_list(self, values):
        """
        Update the results from a list of values.

        :param values: List of metric values
        """
        i = 0
        for label in self.results:
            for metric in self.metrics:
                self.results[label][metric] = values[i]
                i += 1

class ResultsTracker(BaseTracker):
    def __init__(self):
        """Initialize the results tracker with default values."""
        super().__init__()

    def update(self, label, ans1, ans2):
        """
        Update the results based on the provided label and answers.

        :param label: The true label of the result
        :param ans1: The first answer
        :param ans2: The second answer
        :return: The updated result label
        """
        label = label.strip().lower()
        if label not in self.labels:
            raise ValueError(f"Incorrect label: {label}")

        return self._update_result(label, ans1, ans2)

    def _update_result(self, true_label, ans1, ans2):
        """
        Update the result for a specific label.

        :param true_label: The true label
        :param ans1: The first answer
        :param ans2: The second answer
        :return: The updated result label
        """
        outcomes = {
            (1, 0): 'tp',
            (0, 1): 'fp',
            (1, 1): 'tp',
            (0, 0): 'fp'
        }
        if (ans1, ans2) in outcomes:
            self.results[true_label][outcomes[(ans1, ans2)]] += 1
            if (ans1, ans2) == (0, 1):
                self.results[self._opposite_label(true_label)]['fn'] += 1
            if (ans1, ans2) == (0, 0):
                self.results[self._opposite_label(true_label)]['fn'] += 1
        return true_label

    @staticmethod
    def _opposite_label(label):
        """
        Get the opposite label for updating false negatives.

        :param label: The true label
        :return: The opposite label
        """
        opposites = {
            'before': 'after',
            'after': 'before',
            'equal': 'equal',
            'vague': 'equal'
        }
        return opposites[label]

class BaselineResultsTracker(BaseTracker):
    def __init__(self):
        """Initialize the baseline results tracker with default values."""
        super().__init__()

    def update(self, pred, label):
        """
        Update the baseline results based on the prediction and label.

        :param pred: The predicted label index
        :param label: The true label
        :return: The updated result label
        """
        label = label.strip().lower()
        if label not in self.labels:
            raise ValueError(f"Incorrect label: {label}")

        pred_labels = ['before', 'after', 'vague', 'equal']
        true_label = pred_labels[pred]
        if label == true_label:
            self.results[true_label]['tp'] += 1
        else:
            self.results[true_label]['fp'] += 1
            self.results[label]['fn'] += 1

        return true_label


def get_label(question_name, label):
    """
    Get the numerical label for a given question and label.

    :param question_name: The name of the question
    :param label: The label string
    :return: The numerical label
    """
    label = label.strip().lower()
    mapping = {
        'question_1': {'before': 1, 'after': 0, 'vague': 1, 'equal': 0},
        'question_2': {'before': 0, 'after': 1, 'vague': 1, 'equal': 0},
    }
    return mapping.get(question_name, {}).get(label)


def get_label_for_baseline(label):
    """
    Get the numerical label for baseline comparison.

    :param label: The label string
    :return: The numerical label as a list
    """
    label = label.strip().lower()
    mapping = {'before': 0, 'after': 1, 'vague': 2, 'equal': 3}
    return [mapping.get(label)]


def all_equal(list_of_mp_values):
    """
    Check if all values in a list are equal.

    :param list_of_mp_values: List of values
    :return: True if all values are equal, False otherwise
    """
    return all(val.value == list_of_mp_values[0].value for val in list_of_mp_values)
