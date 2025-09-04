import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple, Optional
from torch import Tensor
import math

class JitLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden, cell):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh

        i_i, i_f, i_g, i_o = x_results.chunk(4, 1)
        h_i, h_f, h_g, h_o = h_results.chunk(4, 1)

        i_gate = torch.sigmoid(i_i + h_i)
        f_gate = torch.sigmoid(i_f + h_f)
        g_gate = torch.tanh(i_g + h_g)
        o_gate = torch.sigmoid(i_o + h_o)

        new_cell = f_gate * cell + i_gate * g_gate
        new_hidden = o_gate * torch.tanh(new_cell)

        return new_hidden, new_cell

# ----------------------------------------------------------------------------------------------------------------------
class JitLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden, cell):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            hidden, cell = self.cell(inputs[i], hidden, cell)
            outputs += [hidden]

        return torch.stack(outputs), hidden, cell

# ----------------------------------------------------------------------------------------------------------------------
class JitLSTM(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bias=True, dropout=0.0):
        super(JitLSTM, self).__init__()
        assert bias
        assert dropout == 0.0

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitLSTMLayer(JitLSTMCell, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitLSTMLayer(JitLSTMCell, input_size, hidden_size)] + 
                                       [JitLSTMLayer(JitLSTMCell, hidden_size, hidden_size)
                                        for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, x, hx=None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        
        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if hx is None:
            h = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)
            c = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h, c = hx

        output_hidden_states = torch.jit.annotate(List[Tensor], [])
        output_cell_states = torch.jit.annotate(List[Tensor], [])

        output = x
        i = 0

        for lstm_layer in self.layers:
            output, hidden, cell = lstm_layer(output, h[i], c[i])
            output_hidden_states += [hidden]
            output_cell_states += [cell]
            i += 1

        # Handle batch_first cases for the output too
        if self.batch_first:
            output = output.permute(1, 0, 2)

        final_hidden = torch.stack(output_hidden_states)
        final_cell = torch.stack(output_cell_states)

        return output, (final_hidden, final_cell)
