import numpy as np
import torch
import cuda_add

# Créer deux tableaux numpy
a = np.random.rand(1).astype(np.float32)
b = np.random.rand(1).astype(np.float32)
c = np.zeros_like(a)

#import pdb; pdb.set_trace()

print("a : ",a)
print("b : ",b)

# Appeler la fonction d'addition sur le GPU
cuda_add.addition(a, b, c)

# Afficher le résultat
print("Result:", c)
