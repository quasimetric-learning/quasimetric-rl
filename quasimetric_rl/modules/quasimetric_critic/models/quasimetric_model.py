from typing import Collection, Tuple, Type

import attrs
import torch
import torch.nn as nn

import torchqmet

from ...utils import MLP, LatentTensor


class InnerProduct(torchqmet.QuasimetricBase):
    """Computes the inner product between two vectors."""

    def __init__(
        self,
        input_size: int,
    ) -> None:
        super().__init__(
            input_size,
            num_components=1,
            guaranteed_quasimetric=False,
            transforms=[],
            reduction="sum",
            discount=None,
        )

    def compute_components(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Inputs:
            x (torch.Tensor): Shape [..., input_size]
            y (torch.Tensor): Shape [..., input_size]

        Output:
            d (torch.Tensor): Shape [..., num_components]
        """
        return torch.sum(x * y, dim=-1, keepdim=True)


class MLPHead(torchqmet.QuasimetricBase):
    """MLP value function head that is not a guaranteed metric."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Collection[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        zero_init_last_fc: bool = False,
    ) -> None:
        super().__init__(
            input_size,
            num_components=1,
            guaranteed_quasimetric=False,
            transforms=[],
            reduction="sum",
            discount=None,
        )
        self.mlp = MLP(
            input_size=2 * input_size,
            output_size=1,
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            zero_init_last_fc=zero_init_last_fc,
        )

    def compute_components(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Inputs:
            x (torch.Tensor): Shape [..., input_size]
            y (torch.Tensor): Shape [..., input_size]

        Output:
            d (torch.Tensor): Shape [..., num_components]
        """
        xy = torch.cat([x, y], dim=-1)
        return self.mlp(xy)


class L2(torchqmet.QuasimetricBase):
    r"""
    This is a *metric* (not quasimetric) that is used for debugging & comparison.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(
            input_size,
            num_components=1,
            guaranteed_quasimetric=True,
            transforms=[],
            reduction="sum",
            discount=None,
        )

    def compute_components(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Inputs:
            x (torch.Tensor): Shape [..., input_size]
            y (torch.Tensor): Shape [..., input_size]

        Output:
            d (torch.Tensor): Shape [..., num_components]
        """
        return (x - y).norm(p=2, dim=-1, keepdim=True)


def create_quasimetric_head_from_spec(spec: str) -> torchqmet.QuasimetricBase:
    # Only two are supported
    #   1. iqe(dim=xxx,components=xxx), Interval Quasimetric Embedding
    #   2. l2(dim=xxx), L2 distance
    #   3. mlp(dim=xxx,hidden_sizes=xxx,activation_fn=xxx,zero_init_last_fc=xxx), MLP head
    #   4. inner_prod(dim=xxx), Inner product

    def iqe(*, dim: int, components: int) -> torchqmet.IQE:
        assert dim % components == 0, "IQE: dim must be divisible by components"
        return torchqmet.IQE(dim, dim // components)

    def l2(*, dim: int) -> L2:
        return L2(dim)

    def mlp(
        *,
        dim: int,
        hidden_sizes: Collection[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        zero_init_last_fc: bool = False,
    ) -> MLPHead:
        return MLPHead(
            input_size=dim,
            hidden_sizes=hidden_sizes,
            activation_fn=activation_fn,
            zero_init_last_fc=zero_init_last_fc,
        )

    def inner_prod(*, dim: int) -> InnerProduct:
        return InnerProduct(dim)

    return eval(spec, dict(iqe=iqe, l2=l2, mlp=mlp, inner_prod=inner_prod), {})


class QuasimetricModel(nn.Module):
    r"""
    (*, input_shape)    (*, input_shape)        Input latents
           |                   |
        [MLP specified by projector_arch]       i.e., apply the same MLP on both inputs
           |                   |
        (*, d_proj)          (*, d_proj)        Projected latents
           +---------+---------+
                     |
    [quasimetric head specified by quasimetric_head_spec]
                     |
                    (*)                         Estimated quasimetric d(x, y)
                     or
                    (*, 2)                      if bidirectional=True
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        projector_arch: Tuple[int, ...] = (512,)
        quasimetric_head_spec: str = "iqe(dim=2048,components=64)"

        def make(self, *, input_size: int) -> "QuasimetricModel":
            return QuasimetricModel(
                input_size=input_size,
                projector_arch=self.projector_arch,
                quasimetric_head_spec=self.quasimetric_head_spec,
            )

    input_size: int
    projector: MLP
    quasimetric_head: torchqmet.QuasimetricBase

    def __init__(
        self,
        *,
        input_size: int,
        projector_arch: Tuple[int, ...],
        quasimetric_head_spec: str,
    ):
        super().__init__()
        self.input_size = input_size
        self.quasimetric_head = create_quasimetric_head_from_spec(
            quasimetric_head_spec
        )
        self.projector = MLP(
            input_size,
            self.quasimetric_head.input_size,
            hidden_sizes=projector_arch,
        )

    def forward(
        self, zx: LatentTensor, zy: LatentTensor, *, bidirectional: bool = False
    ) -> torch.Tensor:
        px = self.projector(zx)  # [B x D]
        py = self.projector(zy)  # [B x D]

        if bidirectional:
            px, py = torch.broadcast_tensors(px, py)
            px, py = torch.stack([px, py], dim=-2), torch.stack(
                [py, px], dim=-2
            )  # [B x 2 x D]

        return self.quasimetric_head(px, py)

    # for type hint
    def __call__(
        self, zx: LatentTensor, zy: LatentTensor, *, bidirectional: bool = False
    ) -> torch.Tensor:
        return super().__call__(zx, zy, bidirectional=bidirectional)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}"
