""" Iterator Algorithms
Collection of iterator algorithms inspired by the algorithm library of C++.
Author: Robert Sharp

"""
import itertools
import operator
import functools
from typing import Callable, Iterator, Iterable, Any, Tuple

__all__ = (
    'accumulate', 'adjacent_difference', 'all_of', 'analytic_continuation',
    'any_of', 'difference', 'exclusive_scan', 'flatten', 'fork', 'generate',
    'generate_n', 'inclusive_scan', 'inner_product', 'intersection', 'iota',
    'min_max', 'none_of', 'partial_sum', 'partition', 'product', 'reduce',
    'symmetric_difference', 'transform', 'transform_reduce', 'transposed_sums',
    'union', 'zip_transform',
)


def inclusive_scan(array: Iterable, init=None) -> Iterator:
    """ Inclusive Scan -> Adjacent Pairs

    DocTests:
    >>> list(inclusive_scan(range(1, 10)))
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    >>> list(inclusive_scan(range(1, 10), 0))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]

    @param array: Iterable to be scanned.
    @return: Iterator of Pairs.
    """
    left, right = itertools.tee(array, 2)
    if init is not None:
        left = itertools.chain((init,), left)
    else:
        _ = next(right)
    return zip(left, right)


def exclusive_scan(array: Iterable, init=None) -> Iterator:
    """ Exclusive Scan -> Adjacent Pairs
    Like inclusive_scan, but:
        Inserts an initial value at the beginning and ignores the last value.

    DocTests:
    >>> list(exclusive_scan(range(1, 10)))
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    >>> list(exclusive_scan(range(1, 10), 0))
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    >>> list(exclusive_scan(range(1, 10), 10))
    [(10, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]

    @param array: Iterable to be scanned.
    @param init: Initial Value.
    @return: Iterator of Pairs.
    """
    left, right = itertools.tee(array, 2)
    left = iter(tuple(left)[:-2])
    if init is not None:
        left = itertools.chain((init, ), left)
    else:
        _ = next(right)
    return zip(left, right)


def partition(array: Iterable, predicate: Callable[[Any], bool]) -> Iterator:
    """ Stable Partition
    Arranges all the elements of a group such that any that return true
        when passed to the predicate will be at the front, and the rest will be
        at the back. The size will not change.

    DocTests:
    >>> list(partition(range(1, 10), is_even))
    [2, 4, 6, 8, 1, 3, 5, 7, 9]
    >>> list(partition(range(1, 10), is_odd))
    [1, 3, 5, 7, 9, 2, 4, 6, 8]

    @param array: Iterable of values to be partitioned.
    @param predicate: Unary functor. F(element) -> bool
    @return: Partitioned Iterator.
    """
    top, bottom = [], []
    for itm in array:
        top.append(itm) if predicate(itm) else bottom.append(itm)
    return iter(top + bottom)


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
    @return: Iterable of adjacent differences.
    """
    """
    transform -> exclusive scan, operator.sub
    """
    return itertools.starmap(lambda x, y: y - x, inclusive_scan(array, 0))


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
    Preforms pairwise multiplication across the iterables,
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


def accumulate(array: Iterable):
    """ Accumulate
    Sums up a range of elements. Same as reduce with operator.add

    DocTests:
    >>> accumulate(range(5))
    10
    >>> accumulate(range(11))
    55

    @param array: Iterable of Values to be summed.
    @return: Sum of Values.
    """
    return sum(array)


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


def transform(array: Iterable, func: Callable) -> Iterator:
    """ Transform
    Similar to map but with a reversed signature.

    DocTests:
    >>> list(transform(range(1, 10), add_one))
    [2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> list(transform(range(1, 10), square))
    [1, 4, 9, 16, 25, 36, 49, 64, 81]

    @param array: Iterable of Values.
    @param func: Unary Functor. F(x) -> Value
    @return: Iterator of transformed Values.
    """
    return map(func, array)


def flatten(maybe: object, *args, _flat: bool = True, **kwargs) -> object:
    """ Flatten Maybe Callable
    Recursively calls the maybe_callable object.
    The first such call passes the arguments provided. F(*args, **kwargs) -> G
    All subsequent calls are made without arguments. G()()()...

    DocTests:
    >>> flatten(square, 8)
    64
    >>> flatten(is_even, 2)
    True
    >>> flatten(is_odd, 2)
    False

    @param maybe: An object that might be callable and might return a callable.
    @param args: Optional Positional arguments for the maybe functor.
    @param _flat: Optional Boolean. Flag to bypass flattening if set to False.
    @param kwargs: Optional Keyword arguments for the maybe functor.
    @return: Object returned from the final call: maybe(*args, **kwargs)()()()...
    """
    if _flat and callable(maybe):
        try:
            return flatten(maybe(*args, **kwargs))
        except TypeError:
            return maybe
    else:
        return maybe


def analytic_continuation(func: Callable, num: int, offset: int = 0) -> int:
    """ AC: Integral Analytic Continuation
    AC extends a function such that negative input is automatically mapped
        to the negative number line with an offset.

    DocTests:
    >>> r_test = range_test(range(0, 10))
    >>> it = [analytic_continuation(random_below, 10) for _ in range(1000)]
    >>> all_of(it, r_test)
    True
    >>> r_test = range_test(range(-9, 1))
    >>> it = [analytic_continuation(random_below, -10) for _ in range(1000)]
    >>> all_of(it, r_test)
    True
    >>> r_test = range_test(range(-10, 0))
    >>> it = [analytic_continuation(random_below, -10, -1) for _ in range(1000)]
    >>> all_of(it, r_test)
    True

    @param func: Callable, should have the following signature: F(N: int) -> int
    @param num: Integer, input for func F(N)
    @param offset: Optional Integer, added to output of negative input -F(-N)+O
    @return: Integer. F(N) or -F(-N)+O for negative N

    Random Below Example:
    Consider Random._randbelow as RB:
        RB takes an int N and return a random int in range [0, N-1]
        RB correctly raises an error if N is zero or less - unacceptable.
        This is the canonical example of when to employ AC...
        AC(RB, 10) -> [0, 9]             Pass-through for positive N
        AC(RB, -10) -> [-9, 0]           Maps to the negative number line
        AC(RB, -10, -1) -> [-10, -1]     Same as above with -1 offset

    Note: when the input is zero the offset is returned - zero by default.
    """
    if num < 0:
        return -func(-num) + offset
    elif num == 0:
        return offset
    else:
        return func(num)


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


def iota(start, stop=None, step=1, stride=1) -> Iterator:
    """ Iota
    Iterator of a given range with stride grouping.

    DocTests:
    >>> list(iota(11))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> list(iota(start=1, stop=11))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> list(iota(start=2, stop=21, step=2))
    [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    >>> list(iota(start=2, stop=21, step=2, stride=2))
    [(2, 4), (6, 8), (10, 12), (14, 16), (18, 20)]

    @param start: Beginning
    @param stop: Ending
    @param step: Stepping
    @param stride: Number of groups.
    @return: Iterator of a given multidimensional range.
    """
    if stop is None:
        start, stop = 0, start
    if stride > 1:
        groups = [iter(range(start, stop, step))] * stride
        return zip(*groups)
    else:
        return iter(range(start, stop, step))


def generate(transformer: Callable, *args, **kwargs):
    """ Generate
    Abstract generator function. Infinite Iterator.

    DocTests:
    >>> counter = itertools.count(1)
    >>> gen = generate(next, counter)
    >>> list(next(gen) for _ in range(10))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @param transformer: Functor.
    @param args: Positional arguments for the functor.
    @param kwargs: Keyword arguments for the functor.
    """
    while True:
        yield transformer(*args, **kwargs)


def generate_n(n: int, transformer: Callable, *args, **kwargs):
    """ Generate N
    Abstract generator function. Finite.

    DocTests:
    >>> counter = itertools.count(1)
    >>> list(generate_n(10, next, counter))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    @param n: Maximum number of elements.
    @param transformer: Functor.
    @param args: Positional arguments for the functor.
    @param kwargs: Keyword arguments for the functor.
    """
    for _ in range(n):
        yield transformer(*args, **kwargs)


def zip_transform(transformer: Callable, *args: Iterable) -> Iterator:
    """ Zip Transform
    The transformer should take the same number of arguments as there are iterators.
    Each iteration will call the transformer on the ith elements.
        F(a[i], b[i], c[i]...) ... for each i.

    DocTests:
    >>> l1 = (0, 1, 2, 3)
    >>> l2 = (8, 7, 6, 5)
    >>> l3 = (1, 1, 1, 1)
    >>> list(zip_transform(add_all, l1, l2, l3))
    [9, 9, 9, 9]

    @param transformer: Functor: F(*args) -> Value
    @param args: Any number of Iterators.
    @return: Iterator of transformed Values.
    """
    return itertools.starmap(transformer, zip(*args))


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


def min_max(array: Iterable) -> Tuple:
    """ Min & Max Element

    DocTests:
    >>> min_max(range(1, 10))
    (1, 9)

    @param array: Iterable of Numeric Values
    @return: Tuple(Minimum, Maximum)
    """
    return min(array), max(array)


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


if __name__ == '__main__':
    import doctest
    from random import randrange


    def is_even(n):
        """ Is Even
        Checks a number to see if it is even.

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


    def add_all(*args):
        """ Add All
        Similar to sum, but takes an arbitrary number of arguments.

        DocTests:
        >>> add_all(1)
        1
        >>> add_all(1, 2)
        3
        >>> add_all(1, 2, 3)
        6
        >>> add_all(1, 2, 3, 4)
        10

        @param args: Numbers to be summed.
        @return: Sum of all arguments.
        """
        return sum(args)


    def range_test(target):
        """ Range Test

        DocTests:
        >>> range_test(range(10))(-1)
        False
        >>> range_test(range(10))(0)
        True
        >>> range_test(range(10))(9)
        True
        >>> range_test(range(10))(10)
        False

        @param target: range to test
        @return: Callable
        """

        def inner(n):
            return n in target

        return inner


    def random_below(n):
        """ Random Below

        DocTests:
        >>> test = range_test(range(0, 10))
        >>> all(test(random_below(10)) for _ in range(1000))
        True

        @param n: Upper limit.
        @return: Returns a random integer in range [0, n-1]
        """
        return randrange(n)


    def add_one(n):
        """ Add One

        DocTests:
        >>> add_one(41)
        42
        >>> add_one(-101)
        -100

        @param n: Number added to one.
        @return: Returns N + 1
        """
        return n + 1


    doctest.testmod(verbose=True)
