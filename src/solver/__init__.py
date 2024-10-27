"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver, PruneDetSolver, SparseDetSolver


from typing import Dict

TASKS: Dict[str, BaseSolver] = {
    'detection': DetSolver,
    'detection_prune': PruneDetSolver,
    'detection_sparse': SparseDetSolver,
}
