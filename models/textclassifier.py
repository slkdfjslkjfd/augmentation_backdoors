from torch import nn


class TextClassificationBigModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationBigModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.fc1(embedded)
        embedded = self.act1(embedded)
        embedded = self.fc2(embedded)
        embedded = self.act2(embedded)
        return self.fc3(embedded)

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim,
                sparse=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        #self.fc.weight.data.uniform_(-initrange, initrange)
        #self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.act1(self.fc1(embedded))
        return self.fc2(embedded)
