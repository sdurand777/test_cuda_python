
import gdb
import gdb.printing

import numpy as np

class TensorAccessorPrinter:
    """Pretty-printer for at::GenericPackedTensorAccessorBase objects."""

    def __init__(self, val):
        self.val = val

#     def to_string(self):
#         # Obtient les attributs de l'objet
#         #data_ptr = self.val['data_']
#         sizes = self.val['sizes_']
#         strides = self.val['strides_']
#
# # # Obtenir la taille du tableau à partir du range
# #         low, high = sizes.type.range()
# #         size_length = high - low + 1  # Calcul de la longueur
# #
# #         print(f"Length of sizes from range : {size_length}")
#
#         # Obtenir la longueur du tableau à partir de sa taille en octets
#         size_length = sizes.type.sizeof // sizes[0].type.sizeof
#
#         print(f"Length of sizes from sizeof : {size_length}")
#         
#         # sizes_list = []
#         # for i in range(size_length):
#         #     sizes_list.append(sizes[i])
#
#         # Créer une liste avec les dimensions du tenseur
#         #sizes_list = [sizes[i] for i in range(size_length)]
#         sizes_list = [int(sizes[i].cast(gdb.lookup_type('int'))) for i in range(size_length)]
#         print(f"sizes_list : {sizes_list}")
#
#         # get data
#         data_ptr = self.val['data_']
#         print("data_prt : ", data_ptr)
#         # Convertir le pointeur en entier pour manipuler l'adresse
#         data_address = int(data_ptr)
#         print("data_address : ", data_address)
#
#         # Calculer le nombre total d'éléments dans le tenseur
#         num_elements = np.prod(sizes_list)
#         print("num_elements : ", num_elements)
#
#         # num_elements = 1
#         # for size in sizes_list:
#         #     num_elements *= size
#         #
#         # print("num_elements : ", num_elements)
#
# # Initialiser un tableau NumPy vide avec les bonnes dimensions
#         matrix = np.zeros((sizes[0], sizes[1]), dtype=float)
#
# # Configurer GDB pour afficher tous les éléments
#         gdb.execute('set print elements 0', to_string=True)
#
# # Parcourir les lignes et les colonnes pour remplir la matrice
#         for i in range(sizes[0]):
#             # Calculer l'adresse du début de la ligne `i`
#             line_address = f'({data_address} + {i * sizes[1] * 4})'  # Supposons que `float` = 4 octets
#             # Récupérer les éléments de la ligne `i`
#             expr = f'*(@global float[{sizes[1]}]*) {line_address}'
#             result = gdb.execute(f'print {expr}', to_string=True)
#             
#             # Extraire les valeurs de la sortie GDB
#             result = result[result.index('{')+1 : result.index('}')]  # Extraire les valeurs entre {}
#             elements = [float(x) for x in result.split(',')]  # Convertir en float
#             
#             # Mettre les éléments dans la ligne correspondante du tableau NumPy
#             matrix[i, :] = elements
#
# # Afficher ou utiliser la matrice
#         print("Tensor \n", matrix)
#
#         return "End"

    def to_string(self):
        # Obtenir les attributs de l'objet
        sizes = self.val['sizes_']
        strides = self.val['strides_']

        # Extraire les dimensions du tenseur en convertissant les gdb.Value en entiers
        size_length = sizes.type.sizeof // sizes[0].type.sizeof
        sizes_list = [int(sizes[i].cast(gdb.lookup_type('int'))) for i in range(size_length)]
        
        print(f"sizes_list : {sizes_list}")

        # Récupérer le pointeur de données et le convertir en adresse entière
        data_ptr = self.val['data_']
        data_address = int(data_ptr)
        print("data_address : ", data_address)

        # Calculer le nombre total d'éléments dans le tenseur
        num_elements = np.prod(sizes_list)
        print("num_elements : ", num_elements)

        # Initialiser un tableau NumPy vide avec les bonnes dimensions
        tensor = np.zeros(sizes_list, dtype=float)

        # Configurer GDB pour afficher tous les éléments
        gdb.execute('set print elements 0', to_string=True)

        # Fonction récursive pour remplir le tenseur
        def fill_tensor(indices, offset):
            if len(indices) == len(sizes_list) - 1:
                # Nous sommes au dernier niveau, il faut récupérer une ligne
                size_at_last_dim = sizes_list[-1]
                line_address = f'({data_address} + {offset})'
                expr = f'*(@global float[{size_at_last_dim}]*) {line_address}'
                result = gdb.execute(f'print {expr}', to_string=True)
                
                # Extraire les valeurs entre les accolades
                result = result[result.index('{')+1 : result.index('}')]
                elements = [float(x) for x in result.split(',')]
                
                # Remplir la dernière dimension du tenseur
                tensor[tuple(indices)] = elements
            else:
                # Parcourir les indices récursivement pour chaque dimension
                stride = strides[len(indices)] * 4  # Taille en octets du float (ajuster si nécessaire)
                for i in range(sizes_list[len(indices)]):
                    fill_tensor(indices + [i], offset + i * stride)

        # Lancer le remplissage du tenseur avec une fonction récursive
        fill_tensor([], 0)

        # Afficher ou utiliser le tenseur
        print("Tensor \n", tensor)

        return "End"




def build_pretty_printer():
    print("build pretty for accessor")
    pp = gdb.printing.RegexpCollectionPrettyPrinter("torch")
    pp.add_printer('TensorAccessor', '^at::GenericPackedTensorAccessorBase<.*>$', TensorAccessorPrinter)
    return pp

gdb.printing.register_pretty_printer(gdb.current_objfile(), build_pretty_printer())
