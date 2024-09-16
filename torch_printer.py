import gdb

class TorchTensorPrinter:
    """Pretty-printer for torch::Tensor objects."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        return "torch::Tensor object"

def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("torch")
    pp.add_printer('Tensor', '^at::Tensor$', TorchTensorPrinter)
    return pp

gdb.printing.register_pretty_printer(gdb.current_objfile(), build_pretty_printer())
