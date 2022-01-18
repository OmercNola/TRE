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

        if size_of_longformer == 'base':
            self.hidden_size = 768

        elif size_of_longformer == 'large':
            self.hidden_size = 1024

        self.out = nn.Linear(self.max_len * self.hidden_size, self.output_size)

        # I'm not using them right now, they're here because the model that
        # trained on Boolq was initialized with these layers.
        self.BN = nn.BatchNorm1d(num_features=self.max_len)
        self.Dropout = nn.Dropout(p=Dropout_prob, inplace=False)


    def forward(self, input_ids, attention_mask):
        Output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # all the hidden states from the last layer:
        last_hidden_state = Output.last_hidden_state

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