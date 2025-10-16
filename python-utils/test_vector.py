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

import chipwhisperer  as cw
import numpy          as np
import params         as p
import struct
import utils          as u
from alive_progress import alive_bar
from time import sleep

## This module contains the routine for the acquisition campaign of the
## side-channe traces.

debugPrint = False

def checkInference(weights, biases, ins, receivedOuts):
  inputSize = p.imgWidth * p.imgHeight

  for n in range(0, p.nNeurons):
    expectedOut = sum(weights[inputSize * n : inputSize * (n + 1)] * ins) + biases[n]
    if expectedOut != receivedOuts[n]:
      print(f"Inference error {n}/{p.nNeurons}:")
      print(f"\tExpected: {expectedOut}")
      print(f"\tReceived: {receivedOuts[n]}")

def collect(scope, target, seedInputs, seedMACPruning, enable = False):
  """ Collect the side-channel waveforms. 

  Args:
    - scope         : the handler to the CW scope.
    - target        : the handler to the CW target board.
    - seedInputs    : the seed to the random generator for the inputs.
    - seedMACPruning: the seed for MACPruning IaPAM and toExecTables.
    - enable        : enable the MACPruning countermeasure.

  Return:
    - waves         : the collected side-channel waveforms.
    - IaPAM         : the used IaPAM.
    - toExecTables  : the used toExecTable.
    - inputs        : the used inputs.
  """

  scope.adc.samples = p.nSamples
  waves = np.zeros(shape = (p.nWaves, p.nSamples), dtype = np.float32)

  ## Prepare test vector's inputs.
  # Divide by 8, the number of inputs processed in a single loop iteration
  rngInputs = np.random.default_rng(seedInputs)
  rngMACPruning = np.random.default_rng(seedMACPruning)
  IaPAM = []

  if (enable):
    IaPAM = rngMACPruning.integers(low = 0, high = 2**8, size = p.imgWidth * p.imgHeight // 8, dtype = np.uint8)
  else:
    IaPAM = np.full(shape = p.imgWidth * p.imgHeight // 8, fill_value = 0xFF, dtype = np.uint8)

  toExecTables = rngMACPruning.integers(low = 0, high = 2**8, size = (p.nWaves, (p.imgWidth * p.imgHeight // 8) * p.nNeurons), dtype = np.uint8)

  # Exclude from toExecTables the important pixels (i.e., pixels already set in IaPAM)
  toExecTables = toExecTables ^ (toExecTables & np.tile(IaPAM, p.nNeurons))

  inputs = rngInputs.integers(low = 0, high = 2**7, size = (p.nWaves, p.imgWidth * p.imgHeight), dtype = np.uint8)

  try:
    # Load the IaPAM on the target
    if debugPrint:
      print(">> Send IaPAM")

    msg = bytearray(IaPAM)
    target.simpleserial_write('c', msg)
    target.simpleserial_wait_ack(timeout = 0)

  except Exception as e:
    print(f"Caught exception {e}.")
    raise

  with alive_bar(p.nWaves) as bar:
    for i in range(0, p.nWaves):
      try:
        # Load the toExecTables on the target
        if debugPrint:
          print(">> Send toExecTable")
        msg = bytearray(toExecTables[i])
        target.simpleserial_write('t', msg)
        target.simpleserial_wait_ack(timeout = 0)

        # Load the inputs on the target
        if debugPrint:
          print(">> Send inputs")
        msg = bytearray(inputs[i])
        target.simpleserial_write('a', msg)
        target.simpleserial_wait_ack(timeout = 0)

        # Execute inference
        if debugPrint:
          print(">> Run inference")
        receivedOuts = np.zeros(shape = p.nNeurons, dtype = np.uint32)

        scope.arm()
        target.simpleserial_write('i', bytearray())
        #for n in range(0, p.nNeurons):
        #  receivedOuts[n] = np.array(struct.unpack('<I', target.simpleserial_read('r', 4, timeout = 0))[0]).view(np.uint32)
        #  receivedOuts[n] = np.array(struct.unpack('<I', target.simpleserial_read('r', 4, timeout = 0))[0])
        target.simpleserial_wait_ack(timeout = 0)
        scope.capture()

        waves[i] = scope.get_last_trace()

        #checkInference(p.weights.astype(np.uint32), p.biases.astype(np.uint32), inputs[i], receivedOuts)

      except Exception as e:
        print(f"Caught exception {e}.")
        raise
      bar()

  return np.asarray(waves, dtype = np.float32), IaPAM, toExecTables, inputs
