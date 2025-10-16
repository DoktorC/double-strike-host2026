# Copyright (C) <2025> Lorenzo Casalino <lorenzo.casalino@inria.fr>, Rubén Salvador <ruben.salvador@inria.fr>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy  as np
import os
import params as p

def reverse(binIaPAM):
  assert len(binIaPAM) % 8 == 0

  revBytes = [ b[::-1] for b in np.split(binIaPAM, len(binIaPAM) // 8) ]

  return np.asarray([ i for b in revBytes for i in b ], dtype = bool)

def bytify(binIaPAM):
  assert len(binIaPAM) % 8 == 0

  pack = lambda x :  sum( [x[i] << i for i in range(0, len(x))] )
  bytified = [ pack(b) for b in np.split(binIaPAM, len(binIaPAM) // 8)]

  return bytified

def binarise(byteList):

  unpack = lambda x :  [ (x >> i) & 1 for i in range(0, 8) ]
  binarised = sum([ unpack(b) for b in byteList ], [])
  return binarised

def hw(x):
  """ Retrive the Hamming Weight of the given scalar input @x@ """
  return bin(x).count('1')
