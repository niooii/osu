from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn.modules.rnn import RNNCellBase

class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell that performs the same operation as nn.LSTMCell but is fully coded in Python.

    .. note::
        This class is implemented without relying on CuDNN, which makes it compatible with :func:`torch.vmap` and :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torchrl.modules.tensordict_module.rnn import LSTMCell
        >>> device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
        >>> B = 2
        >>> N_IN = 10
        >>> N_OUT = 20
        >>> V = 4  # vector size
        >>> lstm_cell = LSTMCell(input_size=N_IN, hidden_size=N_OUT, device=device)

        # single call
        >>> x = torch.randn(B, 10, device=device)
        >>> h0 = torch.zeros(B, 20, device=device)
        >>> c0 = torch.zeros(B, 20, device=device)
        >>> with torch.no_grad():
        ...     (h1, c1) = lstm_cell(x, (h0, c0))

        # vectorised call - not possible with nn.LSTMCell
        >>> def call_lstm(x, h, c):
        ...     h_out, c_out = lstm_cell(x, (h, c))
        ...     return h_out, c_out
        >>> batched_call = torch.vmap(call_lstm)
        >>> x = torch.randn(V, B, 10, device=device)
        >>> h0 = torch.zeros(V, B, 20, device=device)
        >>> c0 = torch.zeros(V, B, 20, device=device)
        >>> with torch.no_grad():
        ...     (h1, c1) = batched_call(x, h0, c0)
    """

    __doc__ += nn.LSTMCell.__doc__

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"LSTMCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None:
            for idx, value in enumerate(hx):
                if value.dim() not in (1, 2):
                    raise ValueError(
                        f"LSTMCell: Expected hx[{idx}] to be 1D or 2D, got {value.dim()}D instead"
                    )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = self.lstm_cell(input, hx[0], hx[1])

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret

    def lstm_cell(self, x, hx, cx):
        x = x.view(-1, x.size(1))

        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(
            hx, self.weight_hh, self.bias_hh
        )

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        i_gate = i_gate.sigmoid()
        f_gate = f_gate.sigmoid()
        g_gate = g_gate.tanh()
        o_gate = o_gate.sigmoid()

        cy = cx * f_gate + i_gate * g_gate

        hy = o_gate * cy.tanh()

        return hy, cy


# copy LSTM
class LSTMBase(nn.RNNBase):
    """A Base module for LSTM. Inheriting from LSTMBase enables compatibility with torch.compile."""

    def __init__(self, *args, **kwargs):
        return super().__init__("LSTM", *args, **kwargs)


for attr in nn.LSTM.__dict__:
    if attr != "__init__":
        setattr(LSTMBase, attr, getattr(nn.LSTM, attr))


class LSTM(LSTMBase):
    """A PyTorch module for executing multiple steps of a multi-layer LSTM. The module behaves exactly like :class:`torch.nn.LSTM`, but this implementation is exclusively coded in Python.

    .. note::
        This class is implemented without relying on CuDNN, which makes it compatible with :func:`torch.vmap` and :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torchrl.modules.tensordict_module.rnn import LSTM

        >>> device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
        >>> B = 2
        >>> T = 4
        >>> N_IN = 10
        >>> N_OUT = 20
        >>> N_LAYERS = 2
        >>> V = 4  # vector size
        >>> lstm = LSTM(
        ...     input_size=N_IN,
        ...     hidden_size=N_OUT,
        ...     device=device,
        ...     num_layers=N_LAYERS,
        ... )

        # single call
        >>> x = torch.randn(B, T, N_IN, device=device)
        >>> h0 = torch.zeros(N_LAYERS, B, N_OUT, device=device)
        >>> c0 = torch.zeros(N_LAYERS, B, N_OUT, device=device)
        >>> with torch.no_grad():
        ...     h1, c1 = lstm(x, (h0, c0))

        # vectorised call - not possible with nn.LSTM
        >>> def call_lstm(x, h, c):
        ...     h_out, c_out = lstm(x, (h, c))
        ...     return h_out, c_out
        >>> batched_call = torch.vmap(call_lstm)
        >>> x = torch.randn(V, B, T, 10, device=device)
        >>> h0 = torch.zeros(V, N_LAYERS, B, N_OUT, device=device)
        >>> c0 = torch.zeros(V, N_LAYERS, B, N_OUT, device=device)
        >>> with torch.no_grad():
        ...     h1, c1 = batched_call(x, h0, c0)
    """

    __doc__ += nn.LSTM.__doc__

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: float = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:

        if bidirectional is True:
            raise NotImplementedError(
                "Bidirectional LSTMs are not supported yet in this implementation."
            )

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _lstm_cell(x, hx, cx, weight_ih, bias_ih, weight_hh, bias_hh):

        gates = F.linear(x, weight_ih, bias_ih) + F.linear(hx, weight_hh, bias_hh)

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        i_gate = i_gate.sigmoid()
        f_gate = f_gate.sigmoid()
        g_gate = g_gate.tanh()
        o_gate = o_gate.sigmoid()

        cy = cx * f_gate + i_gate * g_gate

        hy = o_gate * cy.tanh()

        return hy, cy

    def _lstm(self, x, hx):

        h_t, c_t = hx
        h_t, c_t = h_t.unbind(0), c_t.unbind(0)

        outputs = []

        weight_ihs = []
        weight_hhs = []
        bias_ihs = []
        bias_hhs = []
        for weights in self._all_weights:
            # Retrieve weights
            weight_ihs.append(getattr(self, weights[0]))
            weight_hhs.append(getattr(self, weights[1]))
            if self.bias:
                bias_ihs.append(getattr(self, weights[2]))
                bias_hhs.append(getattr(self, weights[3]))
            else:
                bias_ihs.append(None)
                bias_hhs.append(None)

        for x_t in x.unbind(int(self.batch_first)):
            h_t_out = []
            c_t_out = []

            for layer, (
                weight_ih,
                bias_ih,
                weight_hh,
                bias_hh,
                _h_t,
                _c_t,
            ) in enumerate(zip(weight_ihs, bias_ihs, weight_hhs, bias_hhs, h_t, c_t)):
                # Run cell
                _h_t, _c_t = self._lstm_cell(
                    x_t, _h_t, _c_t, weight_ih, bias_ih, weight_hh, bias_hh
                )
                h_t_out.append(_h_t)
                c_t_out.append(_c_t)

                # Apply dropout if in training mode
                if layer < self.num_layers - 1 and self.dropout:
                    x_t = F.dropout(_h_t, p=self.dropout, training=self.training)
                else:  # No dropout after the last layer
                    x_t = _h_t
            h_t = h_t_out
            c_t = c_t_out
            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=int(self.batch_first))

        return outputs, (torch.stack(h_t_out, 0), torch.stack(c_t_out, 0))

    def forward(self, input, hx=None):  # noqa: F811
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        if input.dim() != 3:
            raise ValueError(
                f"LSTM: Expected input to be 3D, got {input.dim()}D instead"
            )
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            h_zeros = torch.zeros(
                self.num_layers,
                max_batch_size,
                real_hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            c_zeros = torch.zeros(
                self.num_layers,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (h_zeros, c_zeros)
        return self._lstm(input, hx)
