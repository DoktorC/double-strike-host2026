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
import matplotlib.patches as patches
import numpy as np

from build_hyps import flatten, rev, split
from corrl      import onePassPearsonCorrl
from functools  import partial
from params     import weights

# MLP implementations considered.
available_implementations = ['unprotected', 'protected', 'circumvented']
# Partition traces according to what weight to skip
available_filteredTypes   = ['exec-5', 'non-exec-5', 'exec-7', 'non-exec-7']

# The analysed implementation
implementation = 'circumvented'

assert implementation in available_implementations

# The available dataset traces for the unprotected implementation
databases_unprotected = [ '01-07-2025-15:20-21'
                        , '01-07-2025-16:32-03'
                        , '01-07-2025-17:47-50'
                        , '01-07-2025-19:30-01'
                        , '01-07-2025-20:44-19']

# The available dataset traces for the protected implementation
databases_protected = [ '01-07-2025-15:55-33'
                      , '01-07-2025-17:06-18'
                      , '01-07-2025-18:21-37'
                      , '01-07-2025-20:03-54'
                      , '02-07-2025-05:41-16']

# The very same set of data of the protected version.
# Yet, the waveforms contain only the important MACs; the
# hypotheses are computed taking into account the non-important MACs.
databases_circumvented = databases_protected

databases = { 'unprotected' : databases_unprotected
            , 'protected'   : databases_protected
            , 'circumvented': databases_circumvented }

firstWaveform = 0
lastWaveform  = 50000

# Parameters set right below for each implementation.
firstSample   = 0
lastSample    = 0
# The length, in samples, of a neuron
neuronLength  = 0
subwaveLength = 0

# Set to 'True' if you want to repeat the experiments in Section V.F.
# True allowed only for circumvented waveforms.
filteredWaveforms = True

# Unprotected -- parameters
if (implementation == 'unprotected'):
  firstSample   = 236
  lastSample    = 18000
  neuronLength  = 3500
  subwaveLength = 1000
  assert filteredWaveforms == False

# Protected -- parameters
# The length and position of the subtrace to analyse depend on the
# number of the skipped/executed non-important MACs.
#
# We use a conservative approach: we iteratively compute the minimum and
# maximum length that the subtrace may have, and we update the position of the
# next subtrace to compute according to these parameters (see lines 177--182).
if (implementation == 'protected'):
  firstSample   = 236
  lastSample    = 18000
  # All the (32 MACs - 17 IMACs) = 15 non-important MACs are skipped (84 samples each)
  neuronMinLength  = 2960
  # All the (32 MACs - 17 IMACs) = 15 non-important MACs are executed (120 samples each)
  neuronMaxLength  = 3500
  # All the 8 MACs are skipped
  subwaveMinLength = 8 * 84
  # All the 8 MACs are executed
  subwaveMaxLength = 8 * 120 + 300
  assert filteredWaveforms == False

# Circumvented -- parameters
if (implementation == 'circumvented'):
  firstSample = 0
  lastSample  = 8500 # Realigned IMACs
  neuronLength = 1700
  # Compiled according to the number of Important MACs given the
  # IaPAM = ['0x7a', '0xf2', '0xaa', '0xd0']
  # Each index provides the beginning for the corresponding set of
  # 8 weights/inputs/MACs.
  subwaveBegins = [0, 500, 1000, 1400]
  # The maximum number of Important MACs for each set of 8 inputs is # 5, given the above IaPAM.
  # Thus, the maximum subwave length to consider is 5 IMACs * 100 samples + 100 samples
  # to take into consideration the next IMAC, which carries additional information on
  # the previously executed MAC (either important or not).
  subwaveLength = 600

  # Choose to partition the trace database according to the exection/skipping
  # of a non-important weight.
  filteredType = 'exec-5'
  assert filteredType in available_filteredTypes
  if filteredWaveforms:
    # This is less than the minimum between the two partitions
    # for all the considered experiments. Between experiments, the minimum
    # number of waveforms may vary. We use this value
    # to provide results based on the same number of waveforms.
    lastWaveform = 24500

# Run the correlation analysis on this number of samples.
windowSize = subwaveLength

# Run the correlation analysis from firstNeuron to the lastNeuron.
firstNeuron = 0
lastNeuron  = 2

# Run the correlation analysis from firstInput to the lastInput
firstInput  = 1
lastInput   = 8
numInputs   = 32

# Store the correlation score each 'corrlSampling' waveforms
corrlSampling = 100

# Reverse the weights.
weights = np.flip(np.split(weights, weights.shape[0] // 8), axis = 1).reshape(-1)

# Compute correlation and ranking on each database.
for db in databases[implementation]:
  wavesPath     = f'../data/{implementation}/waveforms-extract-{db}.npy'
  inpsPath      = f'../data/{implementation}/inputs-extract-{db}.npy'
  accumHypsPath = f'../data/{implementation}/hyps-accum-extract-{db}.npy'

  print(f">> Analysing database {db}")

  w = []
  i = []
  if not filteredWaveforms:
    w = np.load(wavesPath)[firstWaveform:lastWaveform, firstSample:lastSample]

    # reverse the inputs.
    i = np.load(inpsPath)[firstWaveform:lastWaveform, :].astype(np.uint32)
    chunks = list(map(partial(split, nChunks = 4), i))
    i = np.asarray(list(map(flatten, list(map(rev, chunks)))), dtype = np.uint32)

  # Iterate over each neuron.
  for neuron in range(firstNeuron, lastNeuron):
    if filteredWaveforms:
      wavesPath     = f'../data/{implementation}/waveforms-extract-neuron-{neuron}-{filteredType}-{db}.npy'
      inpsPath      = f'../data/{implementation}/inputs-extract-neuron-{neuron}-{filteredType}-{db}.npy'
      accumHypsPath = f'../data/{implementation}/hyps-accum-extract-neuron-{neuron}-{filteredType}-{db}.npy'
      w = np.load(wavesPath)[firstWaveform:lastWaveform, firstSample:lastSample]
      # reverse the inputs.
      i = np.load(inpsPath)[firstWaveform:lastWaveform, :].astype(np.uint32)
      chunks = list(map(partial(split, nChunks = 4), i))
      i = np.asarray(list(map(flatten, list(map(rev, chunks)))), dtype = np.uint32)

    # Iterate over each input/weight/MAC.
    for inputIndex in range(firstInput, lastInput):
      print(f">> Input #{inputIndex} -- Neuron #{neuron}")

      neuronShift = neuron * numInputs
      neuronWaveBegin = neuron * neuronLength

      weightsSet = inputIndex // 8
      weightsSetBegin = weightsSet * 8 + neuronShift
      weightsSetEnd = weightsSetBegin + 8

      inputSetWaveBegin = neuronWaveBegin + ((inputIndex) // 8) * subwaveLength
      inputSetWaveEnd = inputSetWaveBegin + subwaveLength

      if implementation == 'protected':
        neuronMinWaveBegin = neuron * neuronMinLength
        neuronMaxWaveBegin = neuron * neuronMaxLength
        inputSetMaxWaveBegin = neuronMaxWaveBegin + ((inputIndex) // 8) * subwaveMaxLength
        inputSetWaveBegin = neuronMinWaveBegin + ((inputIndex) // 8) * subwaveMinLength
        inputSetWaveEnd = inputSetMaxWaveBegin + subwaveMaxLength

      if implementation == "circumvented":
        neuronWaveBegin = neuron * neuronLength
        inputSetWaveBegin = neuronWaveBegin + subwaveBegins[((inputIndex) // 8)]
        inputSetWaveEnd = inputSetWaveBegin + subwaveLength
        if inputSetWaveEnd > lastSample:
          inputSetWaveEnd = lastSample
        
      trueWeight = weights[weightsSetBegin:weightsSetEnd][inputIndex % 8]

      subwave = w[:, inputSetWaveBegin:inputSetWaveEnd]

      # Load precomputed accumulator's hypotheses for the preceeding weight.
      hyps = np.load(accumHypsPath)[firstWaveform:lastWaveform, neuron, (inputIndex - 1):inputIndex]
      inputs = i[:, inputIndex:inputIndex + 1]

      # Hypotheses computation.
      # Target the accumulator; Hamming Weight leakage model.
      hypsLast = np.asarray([ (np.bitwise_count(hyps + (inputs * weight))[:, 0]) for weight in range(1, 128)], dtype = np.uint32).transpose().astype(np.float32)

      # Compute the Pearson's Correlation Coefficient
      corrls = onePassPearsonCorrl(subwave, hypsLast, corrlSampling, lastWaveform)

      # Transpose to a matrix of dimensions (numSamples, weight candidates)
      corrls = np.absolute(corrls.reshape(corrls.shape[0], hypsLast.shape[1], subwave.shape[1])).transpose((0, 2, 1))
      rankedGuesses = np.asarray([[ np.unique(x, return_index = True) for x in np.flip(np.argsort(sample), axis = 1) ] for sample in corrls], dtype = np.uint8)[:, :, 1, :]
      #Required, as, otherwise, ranking is in the range [0; 128].
      rankedGuesses = rankedGuesses + 1

      savePathCorrls = f'../data/corrls/{implementation}/neuron-{neuron}/input-{inputIndex}/{implementation}-corrls-input-{inputIndex}-neuron-{neuron}-extract-{db}.npy'
      savePathRankings = f'../data/rankings/{implementation}/neuron-{neuron}/input-{inputIndex}/{implementation}-ranking-per-sample-input-{inputIndex}-neuron-{neuron}-extract-{db}.npy'

      if filteredWaveforms:
        savePathCorrls = f'../data/corrls/{implementation}/neuron-{neuron}/input-{inputIndex}/{implementation}-corrls-input-{inputIndex}-neuron-{neuron}-extract-{filteredType}-{db}.npy'
        savePathRankings = f'../data/rankings/{implementation}/neuron-{neuron}/input-{inputIndex}/{implementation}-ranking-per-sample-input-{inputIndex}-neuron-{neuron}-extract-{filteredType}-{db}.npy'

      np.save(savePathCorrls, corrls)
      np.save(savePathRankings, rankedGuesses)

#  """ DEBUG -- Plot correlation score vs samples

#      plotScaleFactor = 2
#      found = False
#      f, a = plt.subplots(1)
#      a.grid(axis = 'both', color = "lightgrey", linewidth = '0.5', linestyle = 'dashed')
#      #Plot the raw waveform
#      #a.plot(w[:, 0] / plotScaleFactor, color = "cornflowerblue", alpha = 0.6)
#      a.plot(np.array(corrls[-1, :, trueWeight - 1]), color = "firebrick", label = "True Weight")
#      a.plot(np.array(np.delete(corrls[-1, :, :], trueWeight - 1, axis = 1)), color = "dimgrey", alpha = 0.5, linestyle = 'dotted')
#      a.set_ylabel('Correlation')
#      a.set_xlabel('Sample')
#      ##Plot the correlation score of the true weight
#      plt.title(label = f"Weight #{inputIndex} - True Weight = {hex(trueWeight)}")
#      #plt.legend()
#      ##plt.savefig(f'../plots/unprotected/neuron-{neuron}/input-{inputIndex}/unprotected-corrls-input-{inputIndex}-neuron-{neuron}' + suffix + '.png')
#      plt.show()
#  """
