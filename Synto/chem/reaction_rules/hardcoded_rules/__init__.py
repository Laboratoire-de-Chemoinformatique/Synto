from .transformations import rules as t_rules
from .decompositions import rules as d_rules


hardcoded_rules = t_rules + d_rules


__all__ = ["hardcoded_rules"]
