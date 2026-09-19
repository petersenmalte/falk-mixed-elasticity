from .adaptive import doerfler_mark, refine_marked
from .convergence import errors_at, loglog_rate
from .domains import create_l_shaped_mesh
from .eigenvalue import assemble_eigenproblem, solve_eigenproblem
from .elements import as_skew, as_stress, falk_function_space
from .estimator import eigenvalue_estimator, global_estimator, solve_estimate
from .manufactured import trigonometric_solution
from .materials import C, C_inv
from .postprocessing import eigenvalue_pair_at, postprocess, postprocessed_eigenvalue
from .problem import solve_source_problem

__all__ = [
    "C",
    "C_inv",
    "falk_function_space",
    "as_stress",
    "as_skew",
    "trigonometric_solution",
    "solve_source_problem",
    "errors_at",
    "loglog_rate",
    "assemble_eigenproblem",
    "solve_eigenproblem",
    "postprocess",
    "postprocessed_eigenvalue",
    "eigenvalue_pair_at",
    "create_l_shaped_mesh",
    "eigenvalue_estimator",
    "global_estimator",
    "solve_estimate",
    "doerfler_mark",
    "refine_marked",
]
