# SPDX-FileCopyrightText: 2024-present Abraham Miller <abraham.r.m@gmail.com>
#
# SPDX-License-Identifier: MIT

import numpy as np
import sympy as sp
from multiprocessing import Pool, cpu_count
from mpmath import mp

def eigs(A, symbols, parallel=True, batch_size=None, pool_size=cpu_count(), staggered=False, base=10):
    """
    Symbolic Eigenvalue Solver (for integer linear eigenvalues).
    Calculates symbolic integer linear eigenvalues for a given compatible matrix.
    Parameters:
       A (2D Matrix): The input symbolic matrix for which eigenvalues are calculated.
                      Matrix must be square and be of a form which can have integer linear eigenvalues.
       symbols (list): The symbols used in the input matrix. Must be of length n (the number of rows/columns).
       parallel (bool): If True, use multiprocessing to calculate the eigenvalues in parallel. Defaults to True.
       batch_size (int): The number of eigenvectors to calculate in each batch. If None, this will be set automatically.
       pool_size (int): The number of processes to use in the multiprocessing pool. Defaults to the number of CPU cores available.
       staggered (bool): If True, use staggered precision to account for larger coefficients. Defaults to False.
       base (int): The base to use for calculations. Defaults to 10.
    Returns:
       (Matrix) The symbolic eigenvalues as linear combinations of the input symbols, 
                each represented by a list of integer coefficients on the symbols from the input matrix.
    """
    base = mp.mpf(base)

    if parallel:
        print(f"Calculating eigenvalues in parallel, using {pool_size} processes.")
        return _eigs_parallel(A, symbols, batch_size, pool_size, staggered, base)
    else:
        return _eigs_synchronous(A, symbols, batch_size, staggered, base)

def _eigs_parallel(A, symbols, batch_size, pool_size, staggered, base):
    """
    Calculates symbolic integer linear eigenvalues for a given compatible matrix in parallel.
    """
    n = np.shape(A)[0]
    assert n == np.shape(A)[1], "Matrix must be square."
    assert len(symbols) == n, "The number of symbols must be equal to the number of rows/columns in the matrix."

    if batch_size is None:
        batch_size = int(np.ceil(n/pool_size))
    
    digits = 2*batch_size if staggered else batch_size
    mp.dps = digits + 10

    num_batches = int(np.ceil(n/batch_size))

    extraction_params = _set_extraction_params(base, digits, staggered)
    dynamics = _set_dynamics(base, n, digits, staggered)
    
    results_by_batch = _calculate_eigenvalue_batches_in_parallel(A, n, batch_size, num_batches, symbols, extraction_params, dynamics, pool_size, staggered, base)
    eigens = _concatenate_batches(results_by_batch, n, batch_size, num_batches)

    annotated_eigens = eigens*np.array([symbols]*n)
    return sp.Matrix(annotated_eigens)

def _eigs_synchronous(A, symbols, batch_size=None, staggered=False, base=10):
    """
    Calculates symbolic integer linear eigenvalues for a given compatible matrix, without multiprocessing.
    """
    n = np.shape(A)[0]
    assert n == np.shape(A)[1], "Matrix must be square."
    assert len(symbols) == n, "The number of symbols must be equal to the number of rows/columns in the matrix."

    if batch_size is None:
        batch_size = n
    
    digits = 2*batch_size if staggered else batch_size
    mp.dps = digits + 10

    num_batches = int(np.ceil(n/batch_size))

    extraction_params = _set_extraction_params(base, digits, staggered)
    dynamics = _set_dynamics(base, n, digits, staggered)
    
    results_by_batch = _calculate_eigenvalue_batches_synchronously(A, n, batch_size, digits, num_batches, symbols, extraction_params, dynamics, staggered, base)
    eigens = _concatenate_batches(results_by_batch, n, batch_size, num_batches)

    annotated_eigens = eigens*np.array([symbols]*n)
    return sp.Matrix(annotated_eigens)

def _set_extraction_params(base, digits, staggered):
    midpoint_value = int(mp.floor(base/2))-1
    if staggered:
        midpoint = sum([mp.mpf(midpoint_value * base ** i) for i in range(-int(np.floor(digits/2)),int(np.ceil(digits/2)))])
    else:
        midpoint = mp.nsum(lambda i: midpoint_value*base**i, [-int(np.floor(digits/2)),int(np.ceil(digits/2)-1)])
    return {
        'midpoint_value': midpoint_value,
        'midpoint': midpoint
    }

def _set_dynamics(base, n, digits, staggered):
    if staggered:
        indicators = [mp.mpf(base ** i) for i in range(-int(np.floor(digits/2)),int(np.ceil(digits/2))-1,2)][::-1]
    else:
        indicators = [mp.mpf(base**i) for i in range(-int(mp.floor(digits/2)),int(mp.ceil(digits/2)))][::-1]

    reals = list(range(n))
    reals_duplicate_check = [1,1]
    while (len(reals_duplicate_check) != len(list(set(reals_duplicate_check)))):
        reals = list(mp.randmatrix(1,n))
        reals_duplicate_check = [mp.nstr(i.real, int(np.floor(digits/2))) for i in reals]
        
    return {'real_values': reals, 'indicator_values': indicators}

def _calculate_eigenvalue_batches_in_parallel(A, n, batch_size, num_batches, symbols, extraction_params, dynamics, pool_size, staggered, base):
    with Pool(processes=pool_size) as pool:
        results = pool.starmap(_calculate_eigenvalues_for_batch, 
                               [(A, n, batch_index, batch_size, batch_size*(2 if staggered else 1), symbols, extraction_params, dynamics, staggered, base) 
                                for batch_index in range(num_batches)])
    return results

def _calculate_eigenvalue_batches_synchronously(A, n, batch_size, digits, num_batches, symbols, extraction_params, dynamics, staggered, base):
    results = [None] * num_batches
    for batch_index in range(num_batches):
        results[batch_index] = _calculate_eigenvalues_for_batch(A, n, batch_index, batch_size, digits, symbols, extraction_params, dynamics, staggered, base)
    return results

def _calculate_eigenvalues_for_batch(A, n, batch_index, batch_size, digits, symbols, extraction_params, dynamics, staggered, base):
    mp.dps = digits + 10
    Ai = _pseudo_symbolic_matrix_segment(A, n, batch_index, batch_size, symbols, dynamics)
    eigenvalues, eigenvectors = mp.eig(Ai)
    eigenvalues, eigenvectors = mp.eig_sort(eigenvalues, eigenvectors) 
    return _extract_symbolic_eigenvalues(eigenvalues, batch_size, digits, extraction_params['midpoint'], extraction_params['midpoint_value'], staggered, base)

def _extract_symbolic_eigenvalues(eigenvalues, batch_size, digits, midpoint, midpoint_value, staggered, base):
    with mp.workdps(digits):
        midpoint_normalized_eigens = [mp.im(i)+midpoint for i in eigenvalues]
        mne_strings = [get_strings(i, digits, midpoint, midpoint_value, base) for i in midpoint_normalized_eigens]
   
    coeffs = [_extract_coef_list(eigenvalue_string, midpoint_value) for eigenvalue_string in mne_strings]
    if staggered:
        coeffs = [[np.sum([i*j for i,j in zip([base,1], x)]) for x in [ci[i:i+2] for i in range(0, digits,2)]] for ci in coeffs]
    return coeffs

def get_strings(ev, digits, midpoint, midpoint_value, base):
    zero_fill = filler_digits(ev, digits, base)

    with mp.workdps(digits):
        esize = digits - zero_fill
        mne_string = mp.nstr(ev, esize, strip_zeros=False, min_fixed=-mp.inf, max_fixed=mp.inf)
        mne_string = '0'*(zero_fill) + mne_string 
        return mne_string

def filler_digits(e, digits, base):
    if (np.abs(e) / (base ** int(np.ceil(digits/2)-1)) >= 1):
        return 0
    j = 2
    while (np.abs(e) / (base ** int(np.ceil(digits/2)-j)) < 1):
        j += 1
    return min(int(np.ceil(digits/2)-1), (j-1))
        
def _extract_coef_list(eigenvalue_string, midpoint_value):
    val = list(eigenvalue_string)
    del val[val.index('.')]
    return [int(i) - midpoint_value for i in val]

def _pseudo_symbolic_matrix_segment(A, n, batch_index, batch_size, symbols, dynamics):
    real_values = dynamics['real_values'].copy()
    indicator_values = dynamics['indicator_values']

    for i in range(batch_index*batch_size, np.min([n,(batch_index+1)*batch_size])):
        real_values[i] = mp.mpc(real_values[i].real, indicator_values[i-batch_index*batch_size])

    subs = {k: v for k, v in zip(symbols, real_values)}
    Af = A.subs(subs)
    return mp.matrix(Af)

def _concatenate_batches(results_by_batch, n, batch_size, num_batches):
    eigens_by_batch = np.array([np.zeros((n,batch_size), dtype=int)]*(num_batches))
    for batch_index in range(num_batches):
        eigens_by_batch[batch_index] = results_by_batch[batch_index]

    return np.array([np.hstack([eigens_by_batch[i,j] for i in range(num_batches)]) for j in range(n)])[0:n,0:n]
