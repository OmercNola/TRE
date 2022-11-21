from __future__ import absolute_import, division, print_function
from torch import nn
from transformers import \
    (AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, BertTokenizer)
from transformers import RobertaTokenizer, AutoConfig, RobertaModel
from utils.utils import count_parameters
from ipdb import set_trace

class Longformer(nn.Module):

    def __init__(
            self,
            longformer_,
            Output_size,
            Dropout_prob,
            size_of_longformer,
            Max_len):
        super().__init__()

        self.model = longformer_

        # Get all of the model's parameters as a list of tuples.
        self.params = list(self.model.named_parameters())

        # print('longformer model has {:} different'
        #       ' named parameters.\n'.format(len(self.params)))
        #
        # print('==== Embedding Layer ====\n')
        # for p in self.params[0:5]:
        #     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        #
        # for i in range(12):
        #     try:
        #         print(f'\n==== {i} Transformer ====\n')
        #
        #         for p in self.params[5:][i * 22:(i * 22) + 22]:
        #             print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        #     except Exception:
        #         print(Exception)
        #
        # print('\n==== Output Layer ====\n')
        # for p in self.params[-2:]:
        #     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        # adjusting layers for training:
        # freeze_model_up_to_this_layer = 0
        # for p in self.params[0:5 + (freeze_model_up_to_this_layer * 22)]:
        #     p[1].requires_grad = False
        #
        # print(f'\n All longformer params are frozen except'
        #       f' {12 - freeze_model_up_to_this_layer} last Transformers,'
        #       f' and Output Layer \n')

        # define the linear layer:

        # size:
        if size_of_longformer == 'base':
            self.hidden_size = 768

        elif size_of_longformer == 'large':
            self.hidden_size = 1024

        # linear layer:
        self.output_size = Output_size
        self.max_len = Max_len
        self.out = nn.Linear(self.max_len * self.hidden_size, self.output_size)

        # I'm not using them right now, they're here because the model that
        # trained on Boolq was initialized with these layers.
        self.BN = nn.BatchNorm1d(num_features=self.max_len)
        self.Dropout = nn.Dropout(p=Dropout_prob, inplace=False)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids:
        :type input_ids:
        :param attention_mask:
        :type attention_mask:
        :return:
        :rtype:
        """

        Output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # all the hidden states from the last layer:
        last_hidden_state = Output.last_hidden_state

        output = self.out(
            last_hidden_state.view(-1, self.max_len * self.hidden_size))

        return output


class Longformer_for_roberta(nn.Module):

    def __init__(
            self,
            roberta,
            Output_size,
            Dropout_prob,
            size_of_longformer,
            Max_len):
        super().__init__()

        self.model = roberta

        # Get all of the model's parameters as a list of tuples.
        self.params = list(self.model.named_parameters())

        # print('longformer model has {:} different'
        #       ' named parameters.\n'.format(len(self.params)))
        #
        # print('==== Embedding Layer ====\n')
        # for p in self.params[0:5]:
        #     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        #
        # for i in range(12):
        #     try:
        #         print(f'\n==== {i} Transformer ====\n')
        #
        #         for p in self.params[5:][i * 22:(i * 22) + 22]:
        #             print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        #     except Exception:
        #         print(Exception)
        #
        # print('\n==== Output Layer ====\n')
        # for p in self.params[-2:]:
        #     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        # adjusting layers for training:
        # freeze_model_up_to_this_layer = 0
        # for p in self.params[0:5 + (freeze_model_up_to_this_layer * 22)]:
        #     p[1].requires_grad = False
        #
        # print(f'\n All longformer params are frozen except'
        #       f' {12 - freeze_model_up_to_this_layer} last Transformers,'
        #       f' and Output Layer \n')

        # define the linear layer:

        # size:
        self.hidden_size = 768

        # linear layer:
        self.output_size = Output_size
        # self.max_len = Max_len
        self.out = nn.Linear(self.hidden_size, self.output_size)

        #self.BN = nn.BatchNorm1d(num_features=self.max_len)
        self.Dropout = nn.Dropout(p=Dropout_prob, inplace=False)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids:
        :type input_ids:
        :param attention_mask:
        :type attention_mask:
        :return:
        :rtype:
        """
        Output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # first hidden state from the last layer:
        last_hidden_state = Output.last_hidden_state[:,0,:]

        output = self.out(
            last_hidden_state.view(-1, self.hidden_size))

        return output




def create_pretrained_model_and_tokenizer(args):
    """
    :return:
    :rtype:
    """

    # get the config:
    config = AutoConfig.from_pretrained(
        "allenai/longformer-base-4096",
        local_files_only=True
    )

    # change longformer dropout prob, default to 0.1:
    config.attention_probs_dropout_prob = args.dropout_p
    config.hidden_dropout_prob = args.dropout_p

    # load tokenizer, add new tokens:
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096",
        local_files_only=True
    )
    special_tokens_dict = {
        'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # load the pretrained longformer model:
    pre_trained_model = AutoModel.from_pretrained(
        "allenai/longformer-base-4096", config=config,
        local_files_only=True
    )
    # print(f'number of model params before adding linear layer: '
    #       f'{count_parameters(pre_trained_model)}')

    # change embeddings size after adding new tokens:
    pre_trained_model.resize_token_embeddings(len(tokenizer))

    model = Longformer(
        pre_trained_model, args.output_size, args.dropout_p,
        args.Size_of_longfor, args.Max_Len
    )

    # print(f'number of model params after adding linear layer: '
    #       f'{count_parameters(model)}')

    return model, tokenizer
# baseline model:


def create_roberta_pretrained_model_and_tokenizer(args):
    """
    :return:
    :rtype:
    """

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    pre_trained_model = RobertaModel.from_pretrained('roberta-base')
    model = Longformer_for_roberta(
        pre_trained_model, args.output_size, args.dropout_p,
        args.Size_of_longfor, args.Max_Len
    )

    print(f'number of model params after adding linear layer: '
          f'{count_parameters(model)}')

    return model, tokenizer



class Longformer_baseline(nn.Module):

    def __init__(self, longformer_, Output_size, size_of_longformer, Max_len):
        super().__init__()

        self.model = longformer_

        # Get all of the model's parameters as a list of tuples.
        self.params = list(self.model.named_parameters())

        # size:
        if size_of_longformer == 'base':
            self.hidden_size = 768

        elif size_of_longformer == 'large':
            self.hidden_size = 1024

        # linear layer:
        self.output_size = Output_size
        self.max_len = Max_len
        self.out = nn.Linear(self.max_len * self.hidden_size, self.output_size)

    def forward(self, input_ids, attention_mask):

        Output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # all the hidden states from the last layer:
        last_hidden_state = Output.last_hidden_state

        output = self.out(
            last_hidden_state.view(-1, self.max_len * self.hidden_size))

        return output


def create_baesline_pretrained_model_and_tokenizer(args):
    """
    :return:
    :rtype:
    """

    # get the config:
    config = AutoConfig.from_pretrained(
        "allenai/longformer-base-4096",
        local_files_only=True
    )

    # change longformer dropout prob, default to 0.1:
    config.attention_probs_dropout_prob = args.dropout_p
    config.hidden_dropout_prob = args.dropout_p

    # load tokenizer, add new tokens:
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/longformer-base-4096",
        local_files_only=True
    )
    special_tokens_dict = {
        'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    # load the pretrained longformer model:
    pre_trained_model = AutoModel.from_pretrained(
        "allenai/longformer-base-4096",
        config=config, local_files_only=True
    )

    # change embeddings size after adding new tokens:
    pre_trained_model.resize_token_embeddings(len(tokenizer))

    # we have 4 lables ('BEFORE', 'AFTER', 'EQUAL', 'VAGUE')
    output_size_for_baseline_classification = 4

    model = Longformer_baseline(
        pre_trained_model,
        output_size_for_baseline_classification,
        args.Size_of_longfor, args.Max_Len
    )

    return model, tokenizer
