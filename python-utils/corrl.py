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

import matplotlib.pyplot  as plt
import numpy              as np

from alive_progress   import alive_bar
from functools        import partial
from math             import ceil

## This module contains the implementation of the one-pass Pearson's correlation
## coefficient.

def deltaMean(sample, mean, numSample):
  return (sample - mean) / numSample

def deltaVar(sample, mean, updatedMean):
  return (sample - mean) * (sample - updatedMean)

def deltaCov(sampleX, sampleY, meanX, meanY, numSample):
  return (sampleX - meanX) * (sampleY - meanY) * (numSample - 1) / numSample

def onePassPearsonCorrl(subwave, hyps, corrlSampling, lastWaveform):
  """
  Compute the One-Pass Pearson's Correlation Coefficient.

  Args:
    - subwave: the portion of side-channel trace to analyse
    - hyps: the leakage hypotheses
    - corrlSampling: the sampling factor of the coefficient
    - lastWaveform: the last waveform to consider

  Returns:
    - A matrix of (lastWaveform / corrlSampling, numSamples * numHyps)
  """

  # Set variables for the one-pass coefficient correlation algorithm.
  cov = np.zeros(shape = (subwave.shape[1] * hyps.shape[1], ))
  meanX = np.tile(subwave[0, :], hyps.shape[1])
  meanY = np.repeat(hyps[0, :], subwave.shape[1])
  varX = np.zeros(shape = (subwave.shape[1] * hyps.shape[1], ))
  varY = np.zeros(shape = (subwave.shape[1] * hyps.shape[1], ))
  corrls = np.zeros(shape = (lastWaveform // corrlSampling, subwave.shape[1] * hyps.shape[1]))

  # One-pass coefficient correlation computation.
  with alive_bar(lastWaveform - 1) as bar:
    for trace in range(1, lastWaveform):
      # array of size numSamples
      _x = np.tile(subwave[trace, :], hyps.shape[1])
      # array of size numCandidates * numSamples
      _y = np.repeat(hyps[trace, :], subwave.shape[1])

      oldMeanX = meanX
      oldMeanY = meanY
      meanX = meanX + deltaMean(_x, oldMeanX, trace + 1)
      meanY = meanY + deltaMean(_y, oldMeanY, trace + 1)
      cov = cov + deltaCov(_x, _y, oldMeanX, oldMeanY, trace + 1)
      varX = varX + deltaVar(_x, oldMeanX, meanX)
      varY = varY + deltaVar(_y, oldMeanY, meanY)

      # Save the correlation score each 'corrlSampling' times.
      if (trace > 0 and ((trace + 1) % corrlSampling) == 0):
        corrls[((trace + 1) // corrlSampling) - 1] = cov/(np.sqrt(varX) * np.sqrt(varY))
      bar()
  return corrls
