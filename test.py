
import torch
import droid_backends  # Le nom de l'extension compilée

a = torch.rand((4, 6, 2), device='cuda')  # Créer des tenseurs sur le GPU
b = torch.ones((4, 6, 2), device='cuda')

test1 = torch.ones((32, 64))  # Créer des tenseurs sur le GPU
test2 = torch.ones((32, 64))



c = droid_backends.depth_filter_d(a, b)  # Appeler la fonction CUDA
print(c)  # Résultat de l'addition
