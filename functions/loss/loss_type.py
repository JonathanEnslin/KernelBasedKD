'''
Sourced from https://github.com/Matthew-Dickson/FilterBasedKnowledgeDistillation/tree/main
'''

from enum import Enum
 
class LossType(Enum):
    FILTER = 1
    TRADITIONAL = 2
    ATTENTION = 3
    CE = 4
    FEATURE = 5
    FILTER_LAST = 6
    FEATURE_LAST = 7