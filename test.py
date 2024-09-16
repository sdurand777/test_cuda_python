
import torch
import droid_backends  # Le nom de l'extension compilée

# a = torch.ones((32, 64), device='cuda')  # Créer des tenseurs sur le GPU
# b = torch.ones((32, 64), device='cuda')

a = torch.ones((32, 64))  # Créer des tenseurs sur le GPU
b = torch.ones((32, 64))



c = droid_backends.depth_filter(a, b)  # Appeler la fonction CUDA
print(c)  # Résultat de l'addition
