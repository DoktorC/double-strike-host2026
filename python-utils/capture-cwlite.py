#! /usr/bin/env python3

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

import chipwhisperer as cw
import json
import numpy as np
import pathlib as pl
import params as p
import sys
import test_vector as tv
import time
import utils as u

from datetime  import datetime
from importlib import reload

## This script implements the REPL interface to run the acquisition campaign of
## side-channel traces.
##
## The script automatically saves the side-channel traces under the path
##                    '{datapath}/waveforms{suffix}.npy'
## where 'datapath' points to some folder in your filesystem (see below) and
## 'suffix' is a unique string generated from the UTC time of running the
## acquisition campaign (see routine 'main()').
##
## The script automatically saves information concerning the acquisition
## campaign:
## * set of important weights/pixels/MACs (IaPAM);
## * set (and their execution order) of each non-important weight/pixel/MAC
##   (toExecTables);
## * set of inputs (inputs);
## * seed used to randomly generate the inputs (seedInputs);
## * seed used to generate the IaPAM and the toExecTables (seedMACPruning);
## * whether MACPruning is enabled or not (enable);
## * Information concerning the CWLite and the target (see routine 'storeExpParams()')
##
## For more information on the REPL interface, please refer to the README.md.

datapath = './data'
fwpath = "./main-CWLITEARM.hex"

def showSplashMsg(target):
  if (not p.isFlashed):
    print("First flash the target.")
    return

  msg = bytearray([])
  # Send 'hello' command.
  target.simpleserial_write('h', msg)

  msgLen = int.from_bytes(target.simpleserial_read('r', 1), 'little')
  msg = target.simpleserial_read('r', msgLen).decode('utf-8')

  print(msg)

def flashTarget(scope):
  try:
    cw.program_target(scope, cw.programmers.STM32FProgrammer, fwpath)
  except Exception as e:
    print(f"Caught exception {e}.")
    raise

def storeWaveforms(waves, suffix):
  np.save(f'{datapath}/waveforms{suffix}', waves)

  print(f"> Saved waveforms in {datapath}")

def storeExpParams(scope, target, IaPAM, toExecTables, inputs, seedInputs, seedMACPruning, enable, suffix):
  np.save(f'{datapath}/IaPAM{suffix}', IaPAM)
  np.save(f'{datapath}/toExecTables{suffix}', toExecTables)

  if np.any(inputs):
    np.save(f'{datapath}/inputs{suffix}', inputs)

  params = dict()
  params['platform'] = "CWLITE"
  params['scope'] = dict()
  params['target'] = dict()
  params['waves'] = dict()

  params['scope']['fw'] = scope.fw_version
  params['scope']['gain'] = { 'mode' : scope.gain.mode
                            , 'gain' : scope.gain.gain
                            , 'db'   : scope.gain.db}
  params['scope']['adc'] =  { 'state'          : scope.adc.state
                            , 'basic_mode'     : scope.adc.basic_mode
                            , 'timeout'        : scope.adc.timeout
                            , 'offset'         : scope.adc.offset
                            , 'presamples'     : scope.adc.presamples
                            , 'decimate'       : scope.adc.decimate
                            , 'fifo_fill_mode' : scope.adc.fifo_fill_mode}

  params['scope']['clock'] = { 'adc_phase'     : scope.clock.adc_phase
                             , 'sampling_rate' : scope.clock.adc_freq
                             , 'freq_ctr'      : scope.clock.freq_ctr
                             , 'freq_ctr_src'  : scope.clock.freq_ctr_src
                             , 'clkgen_src'    : scope.clock.clkgen_src
                             , 'extclk_freq'   : scope.clock.extclk_freq
                             , 'clkgen_mul'    : scope.clock.clkgen_mul
                             , 'clkgen_div'    : scope.clock.clkgen_div
                             , 'target_clock'   : scope.clock.clkgen_freq}

  params['scope']['trigger'] = { 'triggers' : scope.trigger.triggers
                               , 'module'   : scope.trigger.module}

  params['target']['baud'] = target.baud
  params['target']['chip'] = "STM32F303"

  params['waves']['nwaves'] = toExecTables.shape[0]
  params['waves']['nsamples'] = scope.adc.samples
  params['seeds'] = { 'inputs'    : hex(seedInputs)
                    , 'MACPruning': hex(seedMACPruning)}
  params['MACPruning'] = 'enabled' if enable else 'disabled'

  with open(f'{datapath}/params{suffix}.json', 'w') as fp:
    json.dump(params, fp, sort_keys = True, indent = 2)

  print(f"> Saved experimental parameters in {datapath}")

def closeConnection(scope, target):
  scope.dis()
  target.dis()

def openConnection():
  scope = cw.scope()
  target = cw.target(scope, cw.targets.SimpleSerial2)

  scope.default_setup()

  target.baud = 115200
  return scope, target

def main():

  scope, target = openConnection()

  toDis = False
  enable = False
  waves = np.array([])
  IaPAM = np.array([])
  toExecTables = np.array([])
  inputs = np.array([])

  # The seeds used to generate the inputs of the 5 experiments we ran. 
  #seedInputs = 0x7563613458193467198732456879613485760813677658723456
  #seedInputs = 0x4541239724569910347568387465149358761349857619348563
  #seedInputs = 0x9991234576514442349867894567213984765061034589723458
  #seedInputs = 0x8697013450761534789874734532528734561897345619843756
  seedInputs = 0x5598754442222687523134654782222222561999673456123433

  # The seed used to generate the IaPAM and toExecTables
  seedMACPruning = 0x6788796354065713487561380475683745983147599873456154

  while (not toDis):
    print("> Prompt command")
    print("> c -- Collect waveforms")
    print("> d -- Disable MACPruning")
    print("> e -- Enable MACPruning")
    print("> h -- Show splash message")
    print("> l -- Load test vector driver")
    print("> f -- Flash the target")
    print("> q -- Quit")

    try:
      cmd = input('')[0]
    except:
      print("Prompt raised an exception!")
      continue
     
    try:
      if (cmd == 'h'):
        showSplashMsg(target)
      elif (cmd == 'c'):
        waves, IaPAM, toExecTables, inputs = tv.collect(scope, target, seedInputs, seedMACPruning, enable = enable)
        suffix = f'-extract-{datetime.utcnow().strftime("%d-%m-%Y-%H:%M-%S")}'
        storeWaveforms(waves, suffix)
        storeExpParams(scope, target, IaPAM, toExecTables, inputs, seedInputs, seedMACPruning, enable, suffix)
      elif (cmd == 'd'):
        enable = False
        print(f"> MACPruning disabled")
      elif (cmd == 'e'):
        enable = True
        print(f"> MACPruning enabled")
      elif (cmd == 'f'):
        flashTarget(scope)
        target.flush()
        p.isFlashed = True
      elif (cmd == 'l'):
        reload(tv)
        print("> Reloaded test vector")
      elif (cmd == 'q'):
        toDis = True
      else:
        print(f"> Unknown prompt {cmd}")
    except KeyboardInterrupt:
      print("User interrupted script. Continue.")
    except Exception as e:
      print(f"Unhandled exception {e}. Continue.")

  print("Leaving.")
  closeConnection(scope, target)

if __name__ == "__main__":
  main()
