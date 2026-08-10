from contextlib import nullcontext

import torch
from torch import Tensor

from .factories import create_model_and_tokenizer
from utils.utils import load_config


class Encoder:
    def __init__(self, config: str | dict) -> None:
        self.config = load_config(config)

        assert "model_name" in self.config, "Config must contain 'model_name' key."
        self._model, self._tokenizer = create_model_and_tokenizer(self.config["model_name"], self.config)

        self.device = self.config.get("device", "cpu")
        device_type = torch.device(self.device).type
        self.amp_context = (
            torch.amp.autocast(device_type)
            if self.config.get("amp", True)
            else nullcontext()
        )

        self._model.to(self.device)
        self._model.eval()

    def to_device(
        self,
        data: dict[str, Tensor] | tuple[Tensor, ...] | Tensor,
        device: str,
        non_blocking: bool = False,
    ) -> dict[str, Tensor] | tuple[Tensor, ...] | Tensor:
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(data, tuple):
            data = tuple(v.to(device, non_blocking=non_blocking) for v in data)
        elif isinstance(data, Tensor):
            data = data.to(device, non_blocking=non_blocking)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        return data

    def tensorize_doc(
        self,
        documents: list[str],
        pin_memory: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Tokenize documents on CPU, optionally into page-locked memory."""
        ids, masks = self._tokenizer.tensorize_doc(documents)
        if pin_memory and torch.device(self.device).type == "cuda":
            if not ids.is_pinned():
                ids = ids.pin_memory()
            if not masks.is_pinned():
                masks = masks.pin_memory()
        return ids, masks

    def tensorize_qry(
        self,
        queries: list[str],
        pin_memory: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Tokenize queries on CPU, optionally into page-locked memory."""
        ids, masks = self._tokenizer.tensorize_qry(queries)
        if pin_memory and torch.device(self.device).type == "cuda":
            if not ids.is_pinned():
                ids = ids.pin_memory()
            if not masks.is_pinned():
                masks = masks.pin_memory()
        return ids, masks

    def _qry(self, input_ids: Tensor, attention_mask: Tensor, to_cpu: bool = True) -> dict[str, Tensor]:
        with torch.no_grad():
            with self.amp_context:
                Q = self._model.encode_qry(input_ids, attention_mask)

                if to_cpu:
                    self.to_device(Q, "cpu")

                return Q

    def _doc(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        to_cpu: bool = True
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            with self.amp_context:
                D = self._model.encode_doc(input_ids, attention_mask)
                D["mv_repr"] = D["mv_repr"].half()

                if to_cpu:
                    self.to_device(D, "cpu")

                return D

    def encode_qry(
        self,
        queries: list[str],
        bsize: int | None = None,
        to_cpu: bool = False,
    ) -> dict[str, Tensor]:
        tokenized = self.tensorize_qry(queries)
        return self.encode_tokenized_qry(
            tokenized,
            bsize=bsize,
            to_cpu=to_cpu,
        )

    def encode_tokenized_qry(
        self,
        tokenized: tuple[Tensor, Tensor],
        bsize: int | None = None,
        to_cpu: bool = False,
    ) -> dict[str, Tensor]:
        """Encode an already-tokenized query batch.

        Page-locked input tensors are copied asynchronously when the encoder is
        on CUDA.
        """
        bsize = bsize or self.config.get("bsize", None)

        non_blocking = (
            torch.device(self.device).type == "cuda"
            and all(tensor.is_pinned() for tensor in tokenized)
        )
        ids, masks = self.to_device(
            tokenized,
            self.device,
            non_blocking=non_blocking,
        )
        if bsize is not None:
            batches = zip(
                torch.split(ids, bsize, dim=0),
                torch.split(masks, bsize, dim=0)
            )

            all_Q = {}
            for batch in batches:
                Q = self._qry(*batch, to_cpu=to_cpu)

                for k, v in Q.items():
                    all_Q.setdefault(k, []).append(v)

            return {k: torch.cat(v) for k, v in all_Q.items()}

        return self._qry(ids, masks, to_cpu=to_cpu)

    def encode_doc(
        self,
        documents: list[str],
        bsize: int | None = None,
        to_cpu: bool = False,
        flatten: bool = False
    ) -> dict[str, Tensor]:
        tokenized = self.tensorize_doc(documents)
        return self.encode_tokenized_doc(
            tokenized,
            bsize=bsize,
            to_cpu=to_cpu,
            flatten=flatten,
        )

    def encode_tokenized_doc(
        self,
        tokenized: tuple[Tensor, Tensor],
        bsize: int | None = None,
        to_cpu: bool = False,
        flatten: bool = False,
    ) -> dict[str, Tensor]:
        """Encode an already-tokenized document batch.

        Page-locked input tensors are copied asynchronously when the encoder is
        on CUDA, allowing CPU preparation of the following batch to overlap
        with the current batch's GPU work.
        """
        bsize = bsize or self.config.get("bsize", None)

        non_blocking = (
            torch.device(self.device).type == "cuda"
            and all(tensor.is_pinned() for tensor in tokenized)
        )
        ids, masks = self.to_device(
            tokenized,
            self.device,
            non_blocking=non_blocking,
        )
        if bsize is not None:
            batches = zip(
                torch.split(ids, bsize, dim=0),
                torch.split(masks, bsize, dim=0)
            )

            all_D = {}
            for batch in batches:
                D = self._doc(*batch, to_cpu=to_cpu)

                for k, v in D.items():
                    all_D.setdefault(k, []).append(v)

            all_D = {k: torch.cat(v) for k, v in all_D.items()}

        else:
            all_D = self._doc(ids, masks, to_cpu=to_cpu)

        if flatten:
            mv_repr = all_D["mv_repr"]
            mv_mask = all_D["mv_mask"]

            mv_repr = mv_repr.view(-1, mv_repr.size(-1))
            all_D["mv_repr"] = mv_repr[mv_mask.bool().flatten()]
            all_D["lens"] = mv_mask.sum(dim=-1)

            del all_D["mv_mask"]

        return all_D
