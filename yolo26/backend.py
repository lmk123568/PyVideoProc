import json
from collections import OrderedDict, namedtuple

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn

__all__ = ["TRTBackend"]


class TRTBackend(nn.Module):
    def __init__(
        self,
        engine_path,
        device=torch.device("cuda:0"),
    ) -> None:
        super().__init__()

        self.device = device

        # Load TensorRT engine
        self.load_engine(engine_path)

        # Engine context
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT context")

        # Bindings
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False

        io_num = range(self.engine.num_io_tensors)
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

        for i in io_num:
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            if is_input:
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            shape = tuple(self.context.get_tensor_shape(name))

            io = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)

            self.bindings[name] = Binding(name, dtype, shape, io, int(io.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        s = self.bindings["images"].shape
        if im.shape != s:
            raise RuntimeError(f"Input size {im.shape} not equal to engine size {s}")

        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))

        y = [self.bindings[x].data for x in sorted(self.output_names)]
        y = y[0] if len(y) == 1 else [i for i in y]

        return y

    def load_engine(self, engine_path: str) -> None:
        logger = trt.Logger(trt.Logger.INFO)

        # Read file
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            if runtime is None:
                raise RuntimeError("Failed to create TensorRT runtime")
            try:
                meta_len = int.from_bytes(
                    f.read(4), byteorder="little"
                )  # read metadata length
                self.metadata = json.loads(
                    f.read(meta_len).decode("utf-8")
                )  # read metadata
            except UnicodeDecodeError:
                f.seek(0)  # engine file may lack embedded Ultralytics metadata
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
