"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiplication of two floats.

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        Multiplication of x and y

    """
    return x * y


def id(x: float) -> float:
    """Return the input unchanged

    Args:
    ----
        x: a float number.

    Returns:
    -------
        x itself unchanged

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floats

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        Sum of x and y

    """
    return x + y


def neg(x: float) -> float:
    """Return the negative value of a float

    Args:
    ----
        x: a float number.

    Returns:
    -------
        The negative value of x

    """
    return -x


def lt(x: float, y: float) -> float:
    """Check if one float is less than another

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        The boolean result of whether x is less than y

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if one float has equal value with another

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        The boolean result of whether x equals y

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Show which of the two floats is larger

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        The larger one of the two floats

    """
    return x if x >= y else y


def is_close(x: float, y: float) -> float:
    """Show if the two floats are close

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        The boolean result of whether the distance between x and y is less than 1e-2

    """
    return 1.0 if (x - y < 1e-2) and (y - x < 1e-2) else 0.0


def sigmoid(x: float) -> float:
    """Show the sigmoid value of x

    Args:
    ----
        x: a float number.

    Returns:
    -------
        1.0 /(1.0 + e^{-x}) if x >=0 else e^x/(1.0 + e^x)

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))

    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Show the ReLU function of x

    Args:
    ----
        x: a float number.

    Returns:
    -------
        x if x >0 else 0

    """
    if x > 0:
        return x
    else:
        return 0.0


def log(x: float) -> float:
    """Show the natural logarithm of x

    Args:
    ----
        x: a float number.

    Returns:
    -------
        log(e)(x)

    """
    return math.log(x)


def exp(x: float) -> float:
    """Show the exponential funciton of x

    Args:
    ----
        x: a float number.

    Returns:
    -------
        e^x

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Show the reciprocal of x

    Args:
    ----
        x: a float number.

    Returns:
    -------
       1.0 / x

    """
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times of a second float

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        (1 / x) * y

    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second float number

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        (-1 / x ** 2) * y

    """
    return y / -(x**2)


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second float number

    Args:
    ----
        x: a float number.
        y: a float number.

    Returns:
    -------
        1*y if x > 0 and 0 if x < 0

    """
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable

    Args:
    ----
        fn: a function that inputs a float and outputs a float

    Returns:
    -------
        a function that inputs an iterable of floats and outputs an iterable of processed floats

    """

    def apply(inputs: Iterable[float]) -> Iterable[float]:
        results = []
        for i in inputs:
            results.append(fn(i))
        return results

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function

    Args:
    ----
        fn: a function that inputs two floats and outputs a float

    Returns:
    -------
        a function that inputs two iterables of floats and output an iterable of combined floats

    """

    def apply_zip(
        inputs1: Iterable[float], inputs2: Iterable[float]
    ) -> Iterable[float]:
        results = []
        for x, y in zip(inputs1, inputs2):
            results.append(fn(x, y))
        return results

    return apply_zip


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order function that reduces an iterable to a single value using a given function

    Args:
    ----
        fn: a function that combine two values
        start: start value for the combination

    Returns:
    -------
        Function that takes a list of elments and combine elements iteratively using fn

    """

    # TODO: Implement for Task 0.3.
    def apply_reduce(inputs: Iterable[float]) -> float:
        result = start
        for item in inputs:
            result = fn(item, result)
        return result

    return apply_reduce


def negList(input: Iterable[float]) -> Iterable[float]:
    """A function that negates all elements in a list using map

    Args:
    ----
        input: an iterable of floats

    Returns:
    -------
        An iterable of the negative values of all items in the input

    """
    return map(neg)(input)


def addLists(input1: Iterable[float], input2: Iterable[float]) -> Iterable[float]:
    """A function that add corresponding elements from two lists using zipWith

    Args:
    ----
        input1: an iterable of floats
        input2: an iterable of floats

    Returns:
    -------
        An iterable of the sum of elements from the two lists

    """
    return zipWith(add)(input1, input2)


def sum(input: Iterable[float]) -> float:
    """A function that sum all elements in a list using reduce

    Args:
    ----
        input: an iterable of floats

    Returns:
    -------
        a sum of all elements in the list

    """
    return reduce(add, 0.0)(input)


def prod(input: Iterable[float]) -> float:
    """A function that calculate the product of all elements in a list using reduce

    Args:
    ----
        input: an iterable of floats

    Returns:
    -------
        a product of all elements in the list

    """
    return reduce(mul, 1.0)(input)
