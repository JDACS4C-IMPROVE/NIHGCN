# Setup

import os
import candle

file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = None
required = None

class NIHGCN(candle.Benchmark):  # 1
    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions

def initialize_parameters(default_model="class_default_model.txt"):

    # Build benchmark object
    NIHGCN_common = NIHGCN(
        file_path,
        "nihgcn_params.txt",
        "pytorch",
        prog="nihgcn",
        desc="from NIHGCN paper by Peng et al.",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(sctBmk)

    return gParameters
