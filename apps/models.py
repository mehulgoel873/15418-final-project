import sys

sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import math
import numpy as np

np.random.seed(0)


class ConvBN(nn.Module):
    def __init__(self, a, b, k, s, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.sequential = nn.Sequential(
            nn.Conv(
                in_channels=a,
                out_channels=b,
                kernel_size=k,
                stride=s,
            ),
            nn.BatchNorm2d(dim=b),
            nn.ReLU(),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.sequential(x)
        ### END YOUR SOLUTION


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = nn.Sequential(
            *[
                ConvBN(3, 16, 7, 4, device=device, dtype=dtype),
                ConvBN(16, 32, 3, 2, device=device, dtype=dtype),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                ConvBN(32, 64, 3, 2, device=device, dtype=dtype),
                ConvBN(64, 128, 3, 2, device=device, dtype=dtype),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
            ]
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        out = self.conv1(x)
        out = self.conv2(out) + out

        out = self.conv3(out)
        out = self.conv4(out) + out

        out = self.flatten(out)
        out = self.relu(self.linear1(out))
        return self.linear2(out)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        output_size,
        hidden_size,
        num_layers=1,
        seq_model="rnn",
        seq_len=40,
        device=None,
        dtype="float32",
    ):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embed = nn.Embedding(output_size, embedding_size, device, dtype)

        assert seq_model in ("rnn", "lstm")
        if seq_model == "rnn":
            self.seq_model = nn.RNN(
                embedding_size, hidden_size, num_layers, device=device, dtype=dtype
            )
        elif seq_model == "lstm":
            self.seq_model = nn.LSTM(
                embedding_size,
                hidden_size,
                num_layers=num_layers,
                device=device,
                dtype=dtype,
            )
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        T, N = x.shape
        out = self.embed(x)
        # print(out.shape)
        out, h = self.seq_model(out, h)
        # print(out.shape)
        out = self.linear(out.reshape((T * N, -1)))
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True
    )
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32"
    )
    print(cifar10_train_dataset[1][0].shape)
