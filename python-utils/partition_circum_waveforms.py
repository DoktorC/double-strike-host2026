import numpy as np

## This script partition traces captured from implementations with MACPruning
## enabled.
## The script iterate over all the trace datasets specified in 'databases_protected'.
## In each iteration, the script check if a given non-important weight is executed or
## skipped, and places the corresponding trace, inputs and leakage hypotheses in the
## correct partition.
##
## The script partitions the traces, inputs and leakage hypothesese for a given range of
## neurons:
## - firstNeuron: specifies the first neuron (included)
## - lastNeuron: specifies the last neuron (excluded)
##
## To partition according to a specific non-important weight, use the variable 'filteringWeight'.
##
## By default, the scripts saves in numpy format the partitions.
## The 'datapath' variable refers to the base path where to save the partitions.
##
## The script prints on stdout the number of traces in each partition and the minimum between the two partitions.

databases_protected = [ '01-07-2025-15:55-33'
                      , '01-07-2025-17:06-18'
                      , '01-07-2025-18:21-37'
                      , '01-07-2025-20:03-54'
                      , '02-07-2025-05:41-16']

# The weight we use to partition the waveforms.
filteringWeight = 7
firstNeuron = 2
lastNeuron = 5
numInputs = 32

datapath = '../data/circumvented/'

numWaveformsExec = []
numWaveformsNonExec = []

for db in databases_protected: 
  for neuron in range(firstNeuron, lastNeuron):
    waveformsFile = f'{datapath}/IMACs-extract-{db}.npy'
    execMACsFile = f'{datapath}/execMACs-extract-{db}.npy'
    inputsFile = f'{datapath}/inputs-extract-{db}.npy'
    hypsFile = f'{datapath}/hyps-accum-extract-{db}.npy'
    
    waveforms = np.load(waveformsFile)
    execMACs = np.load(execMACsFile)
    inputs = np.load(inputsFile)
    hyps = np.load(hypsFile)
    
    nonExecSet = []
    execSet = []
    
    weightSet = (filteringWeight // 8) + neuron * numInputs // 8

    for i in range(0, waveforms.shape[0]):
      # We subtract from seven since the weights are processed in reverse order.
      if execMACs[i, weightSet] & (0x1 << (7 - filteringWeight)):
        execSet.append(i)
      else:
        nonExecSet.append(i)
 
    numWaveformsExec.append(len(execSet))
    numWaveformsNonExec.append(len(nonExecSet))

    np.save(f'{datapath}/waveforms-extract-neuron-{neuron}-exec-{filteringWeight}-{db}.npy', waveforms[execSet, :])
    np.save(f'{datapath}/inputs-extract-neuron-{neuron}-exec-{filteringWeight}-{db}.npy', inputs[execSet, :])
    np.save(f'{datapath}/hyps-accum-extract-neuron-{neuron}-exec-{filteringWeight}-{db}.npy', hyps[execSet, :])
    
    np.save(f'{datapath}/waveforms-extract-neuron-{neuron}-non-exec-{filteringWeight}-{db}.npy', waveforms[nonExecSet, :])
    np.save(f'{datapath}/inputs-extract-neuron-{neuron}-non-exec-{filteringWeight}-{db}.npy', inputs[nonExecSet, :])
    np.save(f'{datapath}/hyps-accum-extract-neuron-{neuron}-non-exec-{filteringWeight}-{db}.npy', hyps[nonExecSet, :])

print(f"Minimum #waveforms EXEC: {min(numWaveformsExec)}")
print(f"Minimum #waveforms NON-EXEC: {min(numWaveformsNonExec)}")
print(f"Global minimum #waveforms: {min(numWaveformsNonExec + numWaveformsExec)}")
