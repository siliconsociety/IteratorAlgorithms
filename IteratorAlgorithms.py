""" Iterator Algorithms
IA is a collection of iterator algorithms inspired by the C++ algorithms library.
"""
import itertools
import functools
import operator
from typing import Callable, Iterator, Iterable
import doctest

__all__ = (
    'accumulate', 'adjacent_difference', 'all_of', 'any_of', 'difference',
    'exclusive_scan', 'fork', 'generate', 'generate_n', 'inclusive_scan',
    'inner_product', 'intersection', 'iota', 'min_max', 'none_of',
    'partial_sum', 'partition', 'product', 'reduce', 'star_sum', 'star_product',
    'symmetric_difference', 'transform', 'transform_reduce',
    'transposed_sums', 'union', 'zip_transform', 'matrix_multiply',
)


def iota(start, *, stop=None, step=1, stride=0):
    """ Iota
    Iterator of a given range with grouping size equal to the stride.
    If stride is zero - a single dimensional iterator is returned.

    DocTests:
    >>> list(iota(10))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> list(iota(start=1, stop=11))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> list(iota(start=2, stop=21, step=2))
    [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    >>> list(iota(start=2, stop=21, step=2, stride=2))
    [(2, 4), (6, 8), (10, 12), (14, 16), (18, 20)]

    @param start: Beginning. Required.
    @param stop: Ending. Default is None.
    @param step: Stepping. Default is one.
    @param stride: Size of groupings. Default is zero.
    """
    if stop is None:
        start, stop = 0, start
    if stride > 0:
        groups = [iter(range(start, stop, step))] * stride
        yield from zip(*groups)
    else:
        yield from iter(range(start, stop, step))


def generate(func: Callable, *args, **kwargs):
    """ Generate
    Infinite iterator of a callable with arguments.

    DocTests:
    >>> counter = itertools.count(1)
    >>> gen = generate(next, counter)
    >>> list(next(gen) for _ in range(10))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @param func: Callable.
    @param args: Positional arguments for the functor.
    @param kwargs: Keyword arguments for the functor.
    """
    while True:
        yield func(*args, **kwargs)


def generate_n(n: int, func: Callable, *args, **kwargs):
    """ Generate N
    Abstract generator function. Finite.

    DocTests:
    >>> counter = itertools.count(1)
    >>> list(generate_n(10, next, counter))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @param n: Number of elements to generate.
    @param func: Callable.
    @param args: Positional arguments for the functor.
    @param kwargs: Keyword arguments for the functor.
    """
    for _ in range(n):
        yield func(*args, **kwargs)


def fork(array: Iterable, forks: int = 2) -> tuple:
    """ Fork
    Iterator Duplicator. Same as itertools.tee but with a better name.

    DocTests:
    >>> it = iter(range(10))
    >>> a, b, c = fork(it, 3)
    >>> list(c)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> a == b
    False
    >>> list(a) == list(b)
    True

    @param array: Iterable to be forked.
    @param forks: Optional Integer. Default is 2. Represents the number of forks.
    @return: Tuple of N Iterators where N is the number of forks.
    """
    return itertools.tee(array, forks)


def inclusive_scan(array: Iterable, init=None) -> Iterator:
    """ Inclusive Scan -> Adjacent Pairs

    DocTests:
    >>> list(inclusive_scan(range(1, 10)))
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    >>> list(inclusive_scan(range(1, 10), 0))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]

    @param array: Iterable to be scanned.
    @param init: Optional initial value. Default is None.
    @return: Iterator of Pairs.
    """
    left, right = fork(array)
    if init is not None:
        left = itertools.chain((init,), left)
    else:
        _ = next(right)
    return zip(left, right)


def exclusive_scan(array: Iterable, init=None) -> Iterator:
    """ Exclusive Scan -> Adjacent Pairs
    Like inclusive_scan, but ignores the last value.

    DocTests:
    >>> list(exclusive_scan(range(1, 10)))
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    >>> list(exclusive_scan(range(1, 10), 0))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]

    @param array: Iterable to be scanned.
    @param init: Initial Value.
    @return: Iterator of Pairs.
    """
    left, right = fork(array)
    right = iter(tuple(right)[:-1])
    if init is not None:
        left = itertools.chain((init, ), left)
    else:
        _ = next(right)
    return zip(left, right)


def transform(array: Iterable, func: Callable) -> Iterator:
    """ Transform
    Similar to map but with a reversed signature.

    DocTests:
    >>> list(transform(range(10), add_one))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> list(transform(range(10), square))
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    @param array: Iterable of Values.
    @param func: Unary Functor. F(x) -> Value
    @return: Iterator of transformed Values.
    """
    return map(func, array)


def partial_sum(array: Iterable) -> Iterator:
    """ Partial Sum
    Calculates the sum of adjacent pairs.
    This is the opposite of Adjacent Difference.

    DocTests:
    >>> list(partial_sum(range(1, 10)))
    [1, 3, 6, 10, 15, 21, 28, 36, 45]
    >>> list(partial_sum([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @param array: Iterable of Numeric Values.
    @return: Iterator of adjacent sums.
    """
    return itertools.accumulate(array, operator.add)


def adjacent_difference(array: Iterable) -> Iterator:
    """ Adjacent Difference
    Calculates the difference between adjacent pairs.
    This is the opposite of Partial Sum.
    The first iteration compares with zero for proper offset.

    DocTests:
    >>> list(adjacent_difference(range(1, 10)))
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> list(adjacent_difference(partial_sum(range(1, 10))))
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> list(adjacent_difference(partial_sum(range(-10, 11, 2))))
    [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

    @param array: Iterable of Numeric Values.
    @return: Iterator of adjacent differences.
    """
    return itertools.starmap(lambda x, y: y - x, inclusive_scan(array, 0))


def partition(array: Iterable, predicate: Callable) -> Iterator:
    """ Stable Partition
    Arranges all the elements of a group such that any that return true
        when passed to the predicate will be at the front, and the rest will be
        at the back. The size of the output iterator will be the same as the
        size of the input iterable.

    DocTests:
    >>> list(partition(range(1, 10), is_even))
    [2, 4, 6, 8, 1, 3, 5, 7, 9]
    >>> list(partition(range(1, 10), is_odd))
    [1, 3, 5, 7, 9, 2, 4, 6, 8]

    @param array: Iterable of values to be partitioned.
    @param predicate: Unary functor. F(x) -> bool
    @return: Partitioned Iterator.
    """
    top, bottom = [], []
    for itm in array:
        if predicate(itm):
            top.append(itm)
        else:
            bottom.append(itm)
    return itertools.chain(top, bottom)


def reduce(array: Iterable, func: Callable, initial=None):
    """ Reduce
    Similar to accumulate but allows any binary functor and/or an initial value.

    DocTests:
    >>> reduce(range(1, 5), operator.add)
    10
    >>> reduce(range(1, 5), operator.add, 100)
    110
    >>> reduce(range(1, 5), operator.mul)
    24
    >>> reduce(range(1, 5), operator.mul, 0)
    0

    @param array: Iterable of Values to be reduced.
    @param func: Binary Functor.
    @param initial: Initial value. Typically 0 for add or 1 for multiply.
    @return: Reduced Value.
    """
    if initial is not None:
        return functools.reduce(func, array, initial)
    else:
        return functools.reduce(func, array)


def accumulate(array: Iterable):
    """ Accumulate
    Returns the Sum of a range of elements.
        Same as sum() or reduce with operator.add

    DocTests:
    >>> accumulate(range(5))
    10
    >>> accumulate(range(11))
    55

    @param array: Iterable of Values to be summed.
    @return: Sum of Values.
    """
    return sum(array)


def product(array: Iterable):
    """ Product
    Reduce with multiply.
    For counting numbers from 1 to N: returns the factorial of N.

    DocTests:
    >>> product(range(1, 5))
    24
    >>> product(range(5, 10))
    15120

    @param array: Iterable of Values to be reduced.
    @return: Product of all elements multiplied together.
    """
    return reduce(array, operator.mul, initial=1)


def min_max(array: Iterable) -> tuple:
    """ Min & Max Element

    DocTests:
    >>> min_max(range(1, 10))
    (1, 9)
    >>> min_max([100, 42, 69, 1])
    (1, 100)

    @param array: Iterable of Numeric Values
    @return: Tuple(Minimum, Maximum)
    """
    return min(array), max(array)


def star_sum(*args):
    """ Star Sum: Add All Args
    Similar to accumulate, but takes an arbitrary number of arguments.

    DocTests:
    >>> star_sum(1)
    1
    >>> star_sum(1, 2)
    3
    >>> star_sum(1, 2, 3)
    6
    >>> star_sum(1, 2, 3, 4)
    10

    @param args: Numbers to be summed.
    @return: Sum of all arguments.
    """
    return sum(args)


def star_product(*args):
    """ Star Product: Multiply All Args
    Similar to product, but takes an arbitrary number of arguments.

    DocTests:
    >>> star_product(0, 42)
    0
    >>> star_product(3, 3, 3)
    27
    >>> star_product(1, 2, 3, 4)
    24

    @param args: Numbers to be multiplied.
    @return: Product of all arguments.
    """
    return product(args)


def all_of(array: Iterable, predicate: Callable) -> bool:
    """ All of These

    DocTests:
    >>> all_of([], is_even)
    True
    >>> all_of([2, 4, 6], is_even)
    True
    >>> all_of([1, 4, 6], is_even)
    False
    >>> all_of([1, 3, 5], is_even)
    False

    @param array: Iterable to inspect.
    @param predicate: Callable. f(x) -> bool
    @return: Boolean.
    """
    return all(predicate(val) for val in array)


def any_of(array: Iterable, predicate: Callable) -> bool:
    """ Any of These

    DocTests:
    >>> any_of([], is_even)
    False
    >>> any_of([2, 4, 6], is_even)
    True
    >>> any_of([1, 4, 6], is_even)
    True
    >>> any_of([1, 3, 5], is_even)
    False

    @param array: Iterable to inspect.
    @param predicate: Callable. f(x) -> bool
    @return: Boolean.
    """
    return any(predicate(val) for val in array)


def none_of(array: Iterable, predicate: Callable) -> bool:
    """ None Of These

    DocTests:
    >>> none_of([], is_even)
    True
    >>> none_of([2, 4, 6], is_even)
    False
    >>> none_of([1, 4, 6], is_even)
    False
    >>> none_of([1, 3, 5], is_even)
    True

    @param array: Iterable to inspect.
    @param predicate: Callable. f(x) -> bool
    @return: Boolean.
    """
    return not any(predicate(val) for val in array)


def transform_reduce(lhs: Iterable, rhs: Iterable,
                     transformer: Callable, reducer: Callable):
    """ Transform Reduce
    Pairwise transform and then reduction across all results.

    DocTests:
    >>> transform_reduce(range(1, 6), range(1, 6), operator.mul, sum)
    55
    >>> transform_reduce(range(1, 6), range(1, 6), operator.add, product)
    3840

    @param lhs: Left Iterator
    @param rhs: Right Iterator
    @param transformer: Binary Functor F(x, y) -> Value
    @param reducer: Reduction Functor F(Iterable) -> Value
    @return: Reduced Value
    """
    return reducer(itertools.starmap(transformer, zip(lhs, rhs)))


def inner_product(lhs: Iterable, rhs: Iterable):
    """ Inner Product
    Performs pairwise multiplication across the iterables,
        then returns the sum of the products.

    DocTests:
    >>> inner_product(range(1, 6), range(1, 6))
    55
    >>> inner_product(range(11), range(11))
    385

    @param lhs: Left Iterator
    @param rhs: Right Iterator
    @return: Sum of the products.
    """
    return transform_reduce(lhs, rhs, operator.mul, sum)


def matrix_multiply(left, right):
    """ Matrix Product
    Row by Column inner product.

    DocTests
    >>> list(matrix_multiply([[1,2], [3,4]], [[1], [2]]))
    [(5,), (11,)]
    >>> list(matrix_multiply([[10,20], [30,40]], [[10], [20]]))
    [(500,), (1100,)]

    @param left: M x N matrix
    @param right: N x P matrix
    @return: M x P matrix
    """
    right = list(zip(*right))
    return zip(
        inner_product(row, col)
        for row in left
        for col in right
    )


def zip_transform(transducer: Callable, *args: Iterable) -> Iterator:
    """ Zip Transform
    The transducer should take the same number of arguments as the number of
    iterators passed. Each iteration will call the transducer with the ith element
    of each iterable. F(a[i], b[i], c[i]...) ...

    DocTests:
    >>> l1 = (0, 1, 2, 3)
    >>> l2 = (8, 7, 6, 5)
    >>> l3 = (1, 1, 1, 1)
    >>> list(zip_transform(star_sum, []))
    []
    >>> list(zip_transform(star_sum, l1))
    [0, 1, 2, 3]
    >>> list(zip_transform(star_sum, l1, l2))
    [8, 8, 8, 8]
    >>> list(zip_transform(star_sum, l1, l2, l3))
    [9, 9, 9, 9]

    @param transducer: Callable
    @param args: Any number of iterables.
    @return: Iterator of values from the transducer.
    """
    return itertools.starmap(transducer, zip(*args))


def transposed_sums(*args: Iterable) -> Iterator:
    """ Transposed Sums - Column Sums
    The size of the output iterator will be the same as
        the smallest input iterator.

    DocTests:
    >>> l1 = (0, 1, 2, 3)
    >>> l2 = (8, 7, 6, 5)
    >>> l3 = (1, 1, 1, 1)
    >>> list(transposed_sums(l1, l2, l3))
    [9, 9, 9, 9]

    @param args: Arbitrary number of Iterators of numeric values.
    @return: Iterator of transposed sums aka column sums.
    """
    return zip_transform(lambda *a: sum(a), *args)


def union(*args: set) -> set:
    """ Multiple Set Union
    Includes all elements of every set passed in.

    DocTests:
    >>> s1 = {0, 2, 4, 6, 8}
    >>> s2 = {1, 2, 3, 4, 5}
    >>> s3 = {2, 8, 9, 1, 7}
    >>> union(s1, s2, s3)
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    @param args: Arbitrary number of sets.
    @return: Unified set
    """
    return reduce(args, lambda x, y: x | y)


def intersection(*args: set) -> set:
    """ Multiple Set Intersection
    Includes all elements that are common to every set passed in.
    If there is no intersection, it will return the empty set.
    If all sets are the same, it will return the union of all sets.
    Opposite of symmetric_difference.

    DocTests:
    >>> s1 = {0, 2, 4, 6, 8}
    >>> s2 = {1, 2, 3, 4, 5}
    >>> s3 = {2, 8, 9, 1, 7}
    >>> intersection(s1, s2, s3)
    {2}

    @param args: Arbitrary number of sets.
    @return: Set of common elements
    """
    return reduce(args, lambda x, y: x & y)


def difference(*args: set) -> set:
    """ Multiple Set Difference
    Includes every element in the first set that isn't in one of the others.
    If there is no difference, it will return the empty set.

    DocTests:
    >>> s1 = {0, 2, 4, 6, 8}
    >>> s2 = {1, 2, 3, 4, 5}
    >>> s3 = {2, 8, 9, 1, 7}
    >>> difference(s1, s2, s3)
    {0, 6}

    @param args: Arbitrary number of sets.
    @return: Difference between the first set and the rest.
    """
    return reduce(args, lambda x, y: x - y)


def symmetric_difference(*args: set) -> set:
    """ Multiple Set Symmetric Difference
    Includes all elements that are not common to every set passed in.
    If there is no intersection, it will return the union of all sets.
    If all sets are the same, it will return the empty set.
    Opposite of intersection.

    DocTests:
    >>> s1 = {0, 2, 4, 6, 8}
    >>> s2 = {1, 2, 3, 4, 5}
    >>> s3 = {2, 8, 9, 1, 7}
    >>> symmetric_difference(s1, s2, s3)
    {0, 1, 3, 4, 5, 6, 7, 8, 9}

    @param args: Arbitrary number of sets.
    @return: Symmetric difference considering all sets.
    """
    return difference(union(*args), intersection(*args))


# ---- TEST HELPERS ---- #

def is_even(n):
    """ Is Even
    Checks a number to see if it is even.
        Test Helper Function

    DocTests:
    >>> is_even(1)
    False
    >>> is_even(2)
    True
    >>> is_even(3)
    False
    >>> is_even(42)
    True
    >>> is_even(69)
    False

    @param n: Number to be checked
    @return: Boolean
    """
    return n % 2 == 0


def is_odd(n):
    """ Is Odd
    Checks a number to see if it is odd.
        Test Helper Function

    DocTests:
    >>> is_odd(1)
    True
    >>> is_odd(2)
    False
    >>> is_odd(3)
    True
    >>> is_odd(42)
    False
    >>> is_odd(69)
    True

    @param n: Number to be checked
    @return: Boolean
    """
    return n % 2 == 1


def square(n):
    """ Square of N
        Test Helper Function

    DocTests:
    >>> square(1)
    1
    >>> square(4)
    16
    >>> square(-4)
    16

    @param n: Number to be squared.
    @return: Square of n or n * n.
    """
    return n * n


def add_one(n):
    """ Add One
        Test Helper Function

    DocTests:
    >>> add_one(41)
    42
    >>> add_one(-101)
    -100

    @param n: Number added to one.
    @return: Returns N + 1
    """
    return n + 1


if __name__ == '__main__':
    doctest.testmod(verbose=True)
