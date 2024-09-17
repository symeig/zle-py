import numpy as np
import sympy as sp

def validate_eigs(symsol,A,symbols):
    n = A.shape[0]
    for i in range(2):
        rvals=np.random.rand(n)*2-1
        At = np.array(A.subs({k: v for k, v in zip(symbols, rvals)})).astype(np.float64)
        w,v = numeric_eig(At)
        sw = np.array(sp.Matrix(symsol).subs({k: v for k, v in zip(symbols, rvals)})).astype(np.float64)
        sw = np.sort([np.sum(sw[j,:]) for j in range(n)])
        if (np.array_equal(np.round(w,3),np.round(sw,3))==False):
            print(w,'\n\n',sw)
            return False
    return True

def numeric_eig(A):
    eigenvalues, eigenvectors = np.linalg.eig([A])
    eigenvalues = reorder(eigenvalues, eigenvectors)[0][0]
    return eigenvalues, eigenvectors

def reorder(
    eigenvalues, eigenvectors
) -> tuple:
    indices_sort_all = np.argsort(eigenvalues.real)
    for i in range(len(eigenvalues)):
        indices_sort = indices_sort_all[i]

        eigenvalues[i] = eigenvalues[i][indices_sort]
        eigenvectors[i] = eigenvectors[i][:, indices_sort]
    return eigenvalues, eigenvectors
