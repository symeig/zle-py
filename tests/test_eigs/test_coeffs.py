import unittest
from zlepy import eigs
import sympy as sp
import numpy as np
from tests.testing_utils import validate_eigs
import sys

np.set_printoptions(threshold=sys.maxsize)

N = 3
SYMBOLS_03 = sp.symbols('x_0:3')

A_03 = sp.Matrix(
        [
[-4*SYMBOLS_03[0], 0, 0], [0,-4*SYMBOLS_03[2],0], [0,0,-SYMBOLS_03[0]-2*SYMBOLS_03[1]-3*SYMBOLS_03[2]]
        ]
)
A_03s = sp.Matrix(
        [
[-44*SYMBOLS_03[2], 0, 0], [0,-43*SYMBOLS_03[2],0], [0,0,-6*SYMBOLS_03[0]+55*SYMBOLS_03[1]-6*SYMBOLS_03[2]]
        ]
)

class TestEigCoeffs(unittest.TestCase):
    
    def test_eig_with_max_base_case(self):
        result = eigs(A_03, SYMBOLS_03, batch_size=3, parallel=False, staggered=False)
        self.assertTrue(validate_eigs(result, A_03, SYMBOLS_03))

    def test_eig_with_max_staggered(self):
        result = eigs(A_03s, SYMBOLS_03, batch_size=3, parallel=False, staggered=True)
        self.assertTrue(validate_eigs(result, A_03s, SYMBOLS_03))

    def test_eig_parallel_base_case(self):
        result = eigs(A_03, SYMBOLS_03, batch_size=3, parallel=True, staggered=False)
        self.assertTrue(validate_eigs(result, A_03, SYMBOLS_03))

    def test_eig_parallel_staggered(self):
        result = eigs(A_03s, SYMBOLS_03, batch_size=3, parallel=True, staggered=True)
        self.assertTrue(validate_eigs(result, A_03s, SYMBOLS_03))
