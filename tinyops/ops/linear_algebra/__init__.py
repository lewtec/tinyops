"""Linear algebra operations: products, decompositions, matrix properties."""

from .dot_product import dot_product
from .matrix_multiply import matrix_multiply
from .vector_dot_product import vector_dot_product
from .inner_product import inner_product
from .outer_product import outer_product
from .tensor_dot_product import tensor_dot_product
from .einstein_summation import einstein_summation
from .trace import trace
from .diagonal import diagonal
from .kronecker_product import kronecker_product
from .norm import norm
from .determinant import determinant
from .inverse import inverse
from .pseudo_inverse import pseudo_inverse
from .solve_linear_system import solve_linear_system
from .least_squares import least_squares
from .condition_number import condition_number
from .matrix_rank import matrix_rank
from .cholesky_decomposition import cholesky_decomposition
from .qr_decomposition import qr_decomposition
from .matrix_power import matrix_power
