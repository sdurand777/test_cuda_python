
import torch
import droid_backends  # Le nom de l'extension compilée

a = torch.rand((4, 6, 2), device='cuda')  # Créer des tenseurs sur le GPU

# # Taille du tensor
# shape = (4, 6, 2)
#
# # Créer un tensor vide sur le GPU
# a = torch.empty(shape, device='cuda')
#
# # Remplir chaque matrice 6x2 avec des valeurs croissantes (1 pour la première, 2 pour la deuxième, etc.)
# for i in range(shape[0]):
#     a[i] = (i + 1) * torch.ones((shape[1], shape[2]), device='cuda')

b = torch.ones((4, 6, 2), device='cuda')

test1 = torch.ones((32, 64))  # Créer des tenseurs sur le GPU
test2 = torch.ones((32, 64))



c = droid_backends.depth_filter_d(a, b)  # Appeler la fonction CUDA
print(c)  # Résultat de l'addition
