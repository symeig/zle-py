# SPDX-FileCopyrightText: 2024-present Abraham Miller <abraham.r.m@gmail.com>
#
# SPDX-License-Identifier: MIT

import unittest
from zlepy import eigs
from sympy import Symbol, Matrix

class TestEig(unittest.TestCase):
    def test_eig(self):
        matrix = Matrix([[Symbol('a'), Symbol('b')], [Symbol('b'), Symbol('a')]])
        expected = Matrix([[Symbol('a'), -Symbol('b')], [Symbol('a'), Symbol('b')]])
        result = eigs(matrix, [Symbol('a'), Symbol('b')], parallel=False)
        self.assertEqual(result, expected, 'The eig function should return the correct coefficients for each eigenvalue.')

class TestParallelEig(unittest.TestCase):
    def test_parallel_eig(self):
        matrix = Matrix([[Symbol('a'), Symbol('b')], [Symbol('b'), Symbol('a')]])
        expected = Matrix([[Symbol('a'), -Symbol('b')], [Symbol('a'), Symbol('b')]])
        result = eigs(matrix, [Symbol('a'), Symbol('b')], parallel=True)
        self.assertEqual(result, expected, 'The eig function should return the correct coefficients for each eigenvalue.')
