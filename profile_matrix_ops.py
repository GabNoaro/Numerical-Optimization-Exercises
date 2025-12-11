#!/usr/bin/env python3

"""
It is advised to import the whole file instead of individual functions, 
to satisfy all library dependencies. The use of this file is intended 
for notebook usage (no `main()` function).

Code available at:
https://github.com/GabNoaro/Numerical-Optimization-Exercises/blob/ea839d0ca45192554490e8addff93e731881e446/profile_matrix_ops.py

"""

import numpy as np
import time
import gc
import os
import psutil
process = psutil.Process(os.getpid())

print("Code available at 'https://github.com/GabNoaro/Numerical-Optimization-Exercises/blob/ea839d0ca45192554490e8addff93e731881e446/profile_matrix_ops.py'.")

def profile_func(func, size_mult=1000, min_range=1, max_range=11):
    """
    Profile a function that takes a single integer size as input.
    
    For increasing matrix sizes, calls func(size), recording the wall-clock
    runtime and additional memory usage for each size.
    
        Args:
            func (callable):
                Function to profile; must accept a single integer size argument.
            size_mult (int, optional):
                Multiplier mapping loop index i to problem size i * size_mult.
            min_range (int, optional):
                Inclusive starting index for the profiling loop.
            max_range (int, optional):
                Exclusive ending index for the profiling loop.
        
        Returns:
            tuple[np.ndarray, np.ndarray]:
                (resTime, resSpace) arrays with per-size runtime and memory usage.
    
    """
    
    n_points = max_range - min_range
    resTime = np.zeros(n_points)
    resSpace = np.zeros(n_points)

    for i in range(min_range, max_range):
        gc.collect()
        baseRam = process.memory_info().rss
        
        start = time.time()
        size = i * size_mult
        func(size)
        ram = process.memory_info().rss
        end = time.time()

        idx = i - min_range
        resTime[idx] = end - start
        resSpace[idx] = ram - baseRam
    
    print(f"resTime = \n {resTime}")
    print(f"resSpace = \n {resSpace}")

    return resTime, resSpace

def profile_func_custom(
    func, A=None, b=None, size_mult=1000, min_range=1, max_range=11):
    """
    Profile a linear solver with custom matrix/vector inputs.
    
    For each problem size, optionally generates a random linear system,
    copies A and b to avoid in-place side effects, and times a single call
    to func(A_copy, b_copy), tracking runtime and memory usage.
    
    Args:
        func (callable):
            Function to profile; must accept (A, b) as arguments.
        A (np.ndarray, optional):
            Coefficient matrix. If None, a random matrix is generated.
        b (np.ndarray, optional):
            Right-hand side vector. If None, a random vector is generated.
        size_mult (int, optional):
            Multiplier mapping loop index i to problem size i * size_mult
            when generating random systems.
        min_range (int, optional):
            Inclusive starting index for the profiling loop.
        max_range (int, optional):
            Exclusive ending index for the profiling loop.
    
    Returns:
        tuple[np.ndarray, np.ndarray]:
            (resTime, resSpace) arrays with per-size runtime and memory usage.
    
    """
    
    n_points = max_range - min_range
    resTime = np.zeros(n_points)
    resSpace = np.zeros(n_points)

    for i in range(min_range, max_range):
        if A is None or b is None:
            size = i * size_mult
            intA = np.random.randint(-5, 50, (size, size))
            floatA = np.random.rand(size, size)
            A = intA + floatA
            b = np.random.randint(-1, 10, size)

        A_copy = np.copy(A).astype(float)
        b_copy = np.copy(b).astype(float)
        
        gc.collect()
        baseRam = process.memory_info().rss
        
        start = time.time()
        func(A_copy, b_copy)
        ram = process.memory_info().rss
        end = time.time()

        idx = i - min_range
        resTime[idx] = end - start
        resSpace[idx] = ram - baseRam
    
    print(f"resTime = \n{resTime}")
    print(f"\n")
    print(f"resSpace = \n{resSpace}")

    return resTime, resSpace

def runCompareFuncs(func_list, A=None, b=None, size_mult=1000, min_range=1, max_range=11):
    """
    Run profiling for a list of solver functions.
    
    Iterates over the provided functions, printing a header for each and
    calling `profile_func_custom` with shared size and range parameters.
    
    Args:
        func_list (list[callable]):
            List of solver functions, each accepting (A, b).
        A (np.ndarray, optional):
            Fixed coefficient matrix, reused across all profiled functions.
        b (np.ndarray, optional):
            Fixed right-hand side vector, reused across all profiled functions.
        size_mult (int, optional):
            Multiplier mapping loop index i to problem size i * size_mult.
        min_range (int, optional):
            Inclusive starting index for the profiling loop.
        max_range (int, optional):
            Exclusive ending index for the profiling loop.
    
    Returns:
        None
    
    """

    if not isinstance(func_list, list):
        func_list = [func_list]
    
    for func in func_list:
        print(f"\nProfiling `{func.__name__}`:")
        print(f"\n") # Create a space below

        # Call profiler with A and b as fixed arguments
        _, _ = profile_func_custom(
            func,
            A=A,
            b=b,
            size_mult=size_mult,
            min_range=min_range,
            max_range=max_range
        )

def profile_func_generic(func, verbose=True, *args, **kwargs):
    """
    Profile a function recording the wall-clock
    runtime (s) and memory usage (bytes).
    
        Args:
            func (callable):
                Function to profile; must accept a single integer size argument.
            verbose (bool, default:True):
                Whether to return print statements.
            *args, **kwargs:
                Accept profiled function arguments and keyword arguments.
        
        Returns:
            tuple[np.ndarray, np.ndarray]:
                (resTime, resSpace) arrays with runtime (seconds) and memory usage (bytes).
        
    """
    
    gc.collect()
    baseRam = psutil.Process().memory_info().rss

    start = time.perf_counter()
    func(*args, **kwargs)
    end = time.perf_counter()

    ram_now = psutil.Process().memory_info().rss

    resTime = end - start
    resSpace = ram_now - baseRam

    if verbose:
        print(f"resTime = \n {resTime}")
        print(f"\n")
        print(f"resSpace = \n {resSpace}")

    return resTime, resSpace

def runCompareGenericFuncs(func_list, verbose=False, *args, **kwargs):
    """
    Run profiling for a list of functions.
    
    Iterates over the provided functions, printing a header for each and
    calling `profile_func_custom` with shared size and range parameters.
    
    Args:
        func_list (list[callable]):
            List of solver functions, each accepting (A, b).
        verbose (bool, default:False to avoid too many print statements):
                Whether to return print statements.
        *args, **kwargs:
                Accept profiled function arguments and keyword arguments.
    
    Returns:
        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]:
                (resTime_list, resSpace_list) List of np.ndarrays with runtime and memory usage for each funciton.
    
    """

    if not isinstance(func_list, list):
        func_list = [func_list]

    results = []
    for func in func_list:
        if verbose:
            print(f"\nProfiling `{func.__name__}`")
            print(f"\n")
        res = profile_func_generic(func, *args, **kwargs)
        results.append(res)
    return results