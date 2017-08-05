from .RIE import ORAlign1d
from .ARF import MappingRotate

def oralign1d(input, nOrientation=4, return_direction=False):
  return ORAlign1d(nOrientation, return_direction)(input)

def mapping_rotate(input, indices):
  return MappingRotate(indices)(input)
  
__all__ = ["oralign1d", "mapping_rotate"]