from torch import nn
class Longformer(nn.Module):

    def __init__(self, longformer_, Output_size, Dropout_prob, size_of_longformer, Max_len):
        super().__init__()

        # parameters:
        print("creating the longformer module...")

        # define the linear layer:
        self.output_size = Output_size
        self.max_len = Max_len
        self.model = longformer_

        # Get all of the model's parameters as a list of tuples.
        self.params = list(self.model.named_parameters())

        if size_of_longformer == 'base':
            self.hidden_size = 768

            # print('The longformer model has {:} different named parameters.\n'.format(len(self.params)))

            # print('==== Embedding Layer ====\n')
            # for p in self.params[0:5]:
            #     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

            # for i in range(12):
            #     try:
            #         print(f'\n==== {i} Transformer ====\n')
            #
            #         for p in self.params[5:][i * 22:(i * 22) + 22]:
            #             print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
            #     except Exception:
            #         print(Exception)

            # print('\n==== Output Layer ====\n')
            # for p in self.params[-2:]:
            #     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
            #
            # print("")

            # adjusting layers for training:
            # freeze_model_up_to_this_layer = 0
            # for p in self.params[0:5 + (freeze_model_up_to_this_layer * 22)]:
            #     p[1].requires_grad = False

            # print(f'\n All longformer params are frozen except {12 - freeze_model_up_to_this_layer} last Transformers,'
            #       f' and Output Layer \n')

        elif size_of_longformer == 'large':
            self.hidden_size = 1024

            # print('The longformer model has {:} different named parameters.\n'.format(len(self.params)))
            #
            # print('==== Embedding Layer ====\n')
            # for p in self.params[0:5]:
            #     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
            #
            # for i in range(24):
            #     try:
            #         print(f'\n==== {i} Transformer ====\n')
            #
            #         for p in self.params[5:][i*22:(i*22)+22]:
            #             print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
            #     except Exception:
            #         print(Exception)
            #
            # print('\n==== Output Layer ====\n')
            # for p in self.params[-2:]:
            #     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
            #
            # print("")
            #
            # # adjusting layers for training:
            # freeze_model_up_to_this_layer = 0
            # for p in self.params[0:5+(freeze_model_up_to_this_layer*22)]:
            #     p[1].requires_grad = False
            #
            # print(f'\n All longformer params are frozen except {24-freeze_model_up_to_this_layer} last Transformers,'
            #       f' and Output Layer \n')

        self.out = nn.Linear(self.max_len * self.hidden_size, self.output_size)
        self.BN = nn.BatchNorm1d(num_features=self.max_len)
        self.Dropout = nn.Dropout(p=Dropout_prob, inplace=False)


    def forward(self, input_ids, attention_mask):
        Output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # all the hidden states from the last layer:
        last_hidden_state = Output.last_hidden_state

        # last_hidden_state = self.Dropout(last_hidden_state)
        # bn_last_hidden_state = self.BN(last_hidden_state)
        # output = self.out(torch.tanh(bn_last_hidden_state.view(-1, self.max_len * self.hidden_size)))
        # output = torch.log_softmax(res, dim=1)

        output = self.out(last_hidden_state.view(-1, self.max_len * self.hidden_size))

        return output
def create_longformer(model, args):
    """
    :param model:
    :type model:
    :param args:
    :type args:
    :return:
    :rtype:
    """

    model_ = Longformer(
        model, args.output_size, args.dropout_p,
        args.Size_of_longfor, args.Max_Len
    )

    return model_