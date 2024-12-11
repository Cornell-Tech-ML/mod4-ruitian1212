from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_1: List[Any] = list(vals)
    vals_2: List[Any] = list(vals)
    vals_1[arg] = vals_1[arg] + epsilon
    vals_2[arg] = vals_2[arg] - epsilon
    delta: float = f(*vals_1) - f(*vals_2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Compute the derivative from the right end to the current node

        Args:
        ----
            x: value to be accumulated

        Returns:
        -------
            None

        """

    ...

    @property
    def unique_id(self) -> int:
        """A unique identifier for the variable.

        Args:
        ----
            None

        Returns:
        -------
            int: unique identifier

        """
        ...

    def is_leaf(self) -> bool:
        """True if this variable is the leaf node

        Args:
        ----
            None

        Returns:
        -------
            bool: True if the variable is a leaf variable

        """
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant

        Args:
        ----
            None

        Returns:
        -------
            bool: True if the variable is a constant

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of the variable

        Args:
        ----
            None

        Returns:
        -------
            Iterable["Variable"]: The parents of the variable

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Returns the chain rule of the variable

        Args:
        ----
            d_output: The output

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: The chain rule of the variable

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited: List[int] = list()
    final_list: List[Variable] = list()

    def visit(variable: Variable) -> None:
        if variable.is_constant() or variable.unique_id in visited:
            return
        if not variable.is_leaf():
            for parent in variable.parents:
                visit(parent)

        visited.append(variable.unique_id)
        final_list.insert(0, variable)

    visit(variable)
    return final_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    all_variables: Iterable[Variable] = topological_sort(variable)
    all_derivatives = {var.unique_id: 0 for var in all_variables}

    all_derivatives[variable.unique_id] = deriv

    for var in all_variables:
        if var.is_leaf():
            var.accumulate_derivative(all_derivatives[var.unique_id])
        else:
            for parent, deriv in var.chain_rule(all_derivatives[var.unique_id]):
                if parent.is_constant():
                    continue
                else:
                    all_derivatives[parent.unique_id] += deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation.

        Args:
        ----
            *values: The values to be stored.

        Returns:
        -------
            None

        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensor values

        Args:
        ----
            None

        Returns:
        -------
            Tuple[Any, ...]: The saved values.

        """
        return self.saved_values
