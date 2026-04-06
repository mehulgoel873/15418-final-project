"""The module."""

from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return init.ones_like(x, device=x.device) / (1 + ops.exp(-1 * x))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        interval = 1 / np.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                *(input_size, hidden_size),
                low=-1 * interval,
                high=interval,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                *(hidden_size, hidden_size),
                low=-1 * interval,
                high=interval,
                device=device,
                dtype=dtype,
            )
        )
        self.bias_ih = Parameter(
            init.rand(
                *(hidden_size,),
                low=-1 * interval,
                high=interval,
                device=device,
                dtype=dtype,
            )
        )
        self.bias_hh = Parameter(
            init.rand(
                *(hidden_size,),
                low=-1 * interval,
                high=interval,
                device=device,
                dtype=dtype,
            )
        )

        self.bias = bias

        if nonlinearity == "tanh":
            self.nonlinearity = ops.Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = ops.ReLU()
        else:
            assert False, f"Bad nonlinearity {nonlinearity}"
        ### END YOUR SOLUTION

    def forward(self, X_bi, h_bh=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h_bh is None:
            h_bh = init.zeros(
                *(X_bi.shape[0], self.W_hh.shape[0]),
                device=X_bi.device,
                dtype=X_bi.dtype,
            )

        out_bh = X_bi @ self.W_ih
        if self.bias:
            out_bh += ops.broadcast_to(self.bias_ih.reshape((1, -1)), out_bh.shape)

        out_bh += h_bh @ self.W_hh
        if self.bias:
            out_bh += ops.broadcast_to(self.bias_hh.reshape((1, -1)), out_bh.shape)

        return self.nonlinearity(out_bh)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        assert nonlinearity in ("tanh", "relu")
        self.rnn_cells = [
            (
                RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)
                if i == 0
                else RNNCell(
                    hidden_size, hidden_size, bias, nonlinearity, device, dtype
                )
            )
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        output = []
        if h0 is not None:
            h0 = ops.split(h0, axis=0)
        for X_i in ops.split(X, axis=0):
            h_next = []
            for i_layer, rnn_cell in enumerate(self.rnn_cells):
                if h0 is None:
                    h_next.append(rnn_cell(X_i, None))
                else:
                    h_next.append(rnn_cell(X_i, h0[i_layer]))
                X_i = h_next[-1]
            output.append(h_next[-1])
            h0 = h_next
        return ops.stack(output, axis=0), ops.stack(h0, axis=0)

        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        interval = 1 / np.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(
                *(input_size, 4 * hidden_size),
                low=-1 * interval,
                high=interval,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.rand(
                *(hidden_size, 4 * hidden_size),
                low=-1 * interval,
                high=interval,
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    *(4 * hidden_size,),
                    low=-1 * interval,
                    high=interval,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    *(4 * hidden_size,),
                    low=-1 * interval,
                    high=interval,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias_ih = init.zeros(
                *(4 * hidden_size,),
                device=device,
                dtype=dtype,
            )
            self.bias_hh = init.zeros(
                *(4 * hidden_size,),
                device=device,
                dtype=dtype,
            )

        self.sigmoid1 = Sigmoid()
        self.sigmoid2 = Sigmoid()
        self.sigmoid4 = Sigmoid()

        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = (
                init.zeros(
                    *(X.shape[0], self.W_hh.shape[0]), device=X.device, dtype=X.dtype
                ),
                init.zeros(
                    *(X.shape[0], self.W_hh.shape[0]), device=X.device, dtype=X.dtype
                ),
            )

        h0 = h[0]
        c0 = h[1]

        combined = (
            X @ self.W_ih
            + ops.broadcast_to(
                ops.reshape(self.bias_ih, (1, -1)), (X.shape[0], 4 * self.hidden_size)
            )
            + h0 @ self.W_hh
            + ops.broadcast_to(
                ops.reshape(self.bias_hh, (1, -1)), (X.shape[0], 4 * self.hidden_size)
            )
        )
        combined_split = ops.split(combined, axis=1).tuple()
        i = self.sigmoid1(ops.stack(combined_split[: self.hidden_size], axis=1))
        f = self.sigmoid2(
            ops.stack(
                combined_split[self.hidden_size : 2 * self.hidden_size],
                axis=1,
            )
        )
        g = ops.tanh(
            ops.stack(
                combined_split[2 * self.hidden_size : 3 * self.hidden_size],
                axis=1,
            )
        )
        o = self.sigmoid4(ops.stack(combined_split[3 * self.hidden_size :], axis=1))

        c_prime = f * c0 + i * g
        h_prime = o * ops.tanh(c_prime)
        assert c_prime.shape == c0.shape
        assert h_prime.shape == h0.shape

        return h_prime, c_prime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = [
            (
                LSTMCell(input_size, hidden_size, bias, device, dtype)
                if i == 0
                else LSTMCell(hidden_size, hidden_size, bias, device, dtype)
            )
            for i in range(num_layers)
        ]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        output = []
        if h is not None:
            h0 = ops.split(h[0], axis=0)
            c0 = ops.split(h[1], axis=0)
        if h is None:
            h0 = None
            c0 = None
        for X_i in ops.split(X, axis=0):
            h_next = []
            c_next = []
            for i_layer, lstm_cell in enumerate(self.lstm_cells):
                if h0 is None:
                    h_prime = lstm_cell(X_i, None)
                else:
                    h_prime = lstm_cell(X_i, (h0[i_layer], c0[i_layer]))
                h_next.append(h_prime[0])
                c_next.append(h_prime[1])
                X_i = h_next[-1]
            output.append(h_next[-1])
            h0 = h_next
            c0 = c_next
        return ops.stack(output, axis=0), (ops.stack(h0, axis=0), ops.stack(c0, axis=0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.d_embed = embedding_dim
        self.weight = Parameter(
            init.randn(*(num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        T, N = x.shape
        x_onehot = init.one_hot(
            self.num_embeddings, x.reshape((-1,)), device=x.device, dtype=x.dtype
        )
        return (x_onehot @ self.weight).reshape((T, N, -1))
        ### END YOUR SOLUTION
