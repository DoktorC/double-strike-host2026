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

import functools  as fc
import numpy      as np
import params     as p
import utils      as u

from alive_progress import alive_bar

## This script precompute the leakage hypotheses, for each input of each neuron, for:
## - the multiplication
## - the accumulation
## - the final accumulation
## Also, it computes the raw intermeadiate (i.e., leakage model not applied) of
## each accumulation.
##
## The script is parameterised to consider three different implementations
## (listed by the variable 'available_implementations'):
## - unprotected
## - protected
## The implementation to analyse is indicatated by the 'implementation'
## variable.
## The script handle only one dataset at time.
## The integer variable 'dbNumber' is used to select the database to work on
## among the available ones for the given implementation.
##
## The above variables are only used to select the corresponding dataset,
## assuming that the dataset path contains such strings.
## They do not change any other parameters in this script.
##
## By default, the script saves the computed leakage hypotheses and intermediates.
## The datapaths where the script peaks the leakage hypotheses and intermediates
## are defined by 'datapath'.

class Neuron:
  def __init__(self, inputs, weights):
    self.inputs = inputs
    chunks = split(weights, nChunks = 4)
    self.weights = np.asarray(flatten(list(map(lambda x: x[::-1], chunks))), dtype = np.uint32)

  def activate(self):
    self.intermMults = [ i * w for i, w in zip(self.inputs, self.weights) ]
    self.intermAccums = [ sum(self.inputs[0:i] * self.weights[0:i]) for i in range(1, len(self.inputs + 1)) ]
    self.lastAccum = sum(self.intermMults[0:8])

def createNeurons(inputs, weightsChunks):
  return [ Neuron(inputs, weights) for weights in weightsChunks ]

def rev(x):
  return [ y[::-1] for y in x ]

def split(array, nChunks):
  return np.split(array, nChunks)

def flatten(list):
  return [ x for sublist in list for x in sublist ]

datapath='../data'

# MLP implementations considered.
available_implementations = ['unprotected', 'protected']

# What implementation to consider
implementation = 'protected'

assert implementation in available_implementations

# The available dataset traces for the unprotected implementation
databases_unprotected = [ '-extract-01-07-2025-15:20-21'
                        , '-extract-01-07-2025-16:32-03'
                        , '-extract-01-07-2025-17:47-50'
                        , '-extract-01-07-2025-19:30-01'
                        , '-extract-01-07-2025-20:44-19']

# The available dataset traces for the protected implementation
databases_protected = [ '-extract-01-07-2025-15:55-33'
                      , '-extract-01-07-2025-17:06-18'
                      , '-extract-01-07-2025-18:21-37'
                      , '-extract-01-07-2025-20:03-54'
                      , '-extract-02-07-2025-05:41-16']

databases = { 'unprotected' : databases_unprotected
            , 'protected'   : databases_protected }


if __name__ == '__main__':
  dbNumber = 0
  assert dbNumber >= 0 and dbNumber < len(databases[implementation])

  with alive_bar(len(databases[implementation])) as bar:
    for suffix in databases[implementation]:
      print(f'>> Database inputs{suffix}')
      inputs = np.load(f'{datapath}/{implementation}/inputs' + suffix + '.npy').astype(np.uint32)
      weights = p.weights.astype(np.uint32)

      # For each waveform, preserve only the employed weights
      execMACs = np.load(f'{datapath}/{implementation}/execMACs' + suffix + '.npy')
      execMACs = np.asarray([u.binarise(e) for e in execMACs])

      # For standard CPA analysis
      #execMACs = np.full(shape = (inputs.shape[0], weights.shape[0]), fill_value = 0x01, dtype = inputs.dtype)
      weights = weights.reshape((1, weights.shape[0])) * execMACs

      weights = np.asarray(np.split(weights, 5, axis = 1), dtype = np.uint32)

      chunks = list(map(fc.partial(split, nChunks = 4), inputs))
      inputs = np.asarray(list(map(flatten, list(map(rev, chunks)))), dtype = np.uint32)

      neuronsPerWaveform = [ createNeurons(i, weights[:, index, :]) for index, i in enumerate(inputs) ]
      _ = [ [ n.activate() for n in neurons ] for neurons in neuronsPerWaveform ]
 
      hypsAccum = np.asarray([ [ n.intermAccums for n in neurons ] for neurons in neuronsPerWaveform ], dtype = np.uint32)
      leakageHypsAccum = np.asarray([ [ list(map(u.hw, n.intermAccums)) for n in neurons ] for neurons in neuronsPerWaveform ], dtype = np.uint32)
      leakageHypsMult  = np.asarray([ [ list(map(u.hw, n.intermMults)) for n in neurons ] for neurons in neuronsPerWaveform ], dtype = np.uint32)
      leakageHypsLastAccum = np.asarray([ [ u.hw(n.lastAccum) for n in neurons ] for neurons in neuronsPerWaveform ], dtype = np.uint32)

      #np.save(f'{datapath}/{implementation}/leakage-hyps-mult-extract'  + suffix + '.npy', leakageHypsMult)
      #np.save(f'{datapath}/{implementation}/leakage-hyps-accum-extract' + suffix + '.npy', leakageHypsAccum)
      #np.save(f'{datapath}/{implementation}/leakage-hyps-last-accum-extract' + suffix + '.npy', leakageHypsLastAccum)
      np.save(f'{datapath}/circumvented/hyps-accum-extract' + suffix + '.npy', hypsAccum)
      bar()
