
import gdb
import struct

class AtTensorPrinter:
    """Pretty-printer for at::Tensor objects."""

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            # Accéder à l'implémentation du tensor
            impl = self.val['impl_']['target_']
            
            # Accéder aux tailles du tensor
            sizes = impl['sizes_']
            shape = [int(sizes['data_'][i]) for i in range(sizes['size_'])]

            # Accéder aux données du tensor
            storage = impl['storage_']
            data_ptr = storage['data_ptr_']['ptr_']

            # Calculer le nombre d'éléments
            numel = 1
            for dim in shape:
                numel *= dim

            # Lire les données brutes
            float_format = 'f' * numel
            data = gdb.selected_inferior().read_memory(data_ptr, numel * 4)  # 4 bytes per float
            values = struct.unpack(float_format, data)

            # Formatage de la chaîne de sortie
            return f"at::Tensor with shape {shape}, values: {values}"

        except Exception as e:
            return f"Error: {str(e)}"

def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("torch")
    pp.add_printer('Tensor', '^at::Tensor$', AtTensorPrinter)
    return pp

gdb.printing.register_pretty_printer(gdb.current_objfile(), build_pretty_printer())


