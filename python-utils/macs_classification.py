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

import json
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pathlib as pl
import params as p
import sys
import time
import utils as u

from alive_progress import alive_bar
from datetime import datetime
from scared.signal_processing import pattern_detection

## This module contains the routines used to classify the MACs/weights/inputs.
## For more information, check the documentation of each routine.
# The available dataset traces for the protected implementation

databases_protected = [ '01-07-2025-15:55-33'
                      , '01-07-2025-17:06-18'
                      , '01-07-2025-18:21-37'
                      , '01-07-2025-20:03-54'
                      , '02-07-2025-05:41-16']

datapathProt = f"../data/protected"
datapathCirc = f"../data/protected"

# Set font parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\boldmath'

style.use('tableau-colorblind10')

def checkIaPAMs(IaPAMs, refIndex):
  """
  Check if the identified IaPAMs are consistent to a reference IaPAM pattern.
  It returns two arrays: one with consistent IaPAMs, the other with inconsistent
  ones.

  Args:
    - IaPAMs: the set of identified IaPAMs; numpy array of shape (numWaves, imgWidth * imgHeight).
    - refIndex: the index of the IaPAM (in IaPAMs) acting as the reference IaPAM.

  Returns:
    Two numpy arrays, one for consistent IaPAMs, one for inconsistent ones.
  """

  ref = IaPAMs[refIndex]
  consistencies = []
  inconsistencies = []
  for i, IaPAM in enumerate(IaPAMs):
    if not (ref == IaPAM).all():
      inconsistencies.append(i)
    else:
      consistencies.append(i)

  return np.asarray(consistencies, dtype = np.uint32), np.asarray(inconsistencies, dtype = np.uint32)

def saveIaPAM(IaPAM, IMACs, NIMACExecs, orderExecMACs, execMACs, suffix = ""):
  print(">> Extraction terminated")
  print(">> Identified IaPAM, IMACs and exec'd MACs positions")
  print(f">> Saving IaPAM in {datapathCirc}/IaPAM-extract-{suffix}.npy")
  print(f">> Saving IMACs in {datapathCirc}/IMACs-extract-{suffix}.npy")
  print(f">> Saving IMACs in {datapathCirc}/NIMACExecs-extract-{suffix}.npy")
  print(f">> Saving IMACs in {datapathCirc}/orderExecMACs-extract-{suffix}.npy")
  print(f">> Saving exec'd MACs in {datapathCirc}/execMACs-extract-{suffix}.npy")

  np.save(f"{datapathCirc}/IaPAM-extract-{suffix}.npy", IaPAM)
  np.save(f"{datapathCirc}/IMACs-extract-{suffix}.npy", IMACs)
  np.save(f"{datapathCirc}/NIMACExecs-extract-{suffix}.npy", NIMACExecs)
  np.save(f"{datapathCirc}/orderExecMACs-extract-{suffix}.npy", orderExecMACs)
  np.save(f"{datapathCirc}/execMACs-extract-{suffix}.npy", execMACs)

def extractIaPAM(waves, patternIMAC, patternNIMACExec, patternNIMACSkip, suffix = ""):
  """
  Identify and extract the MAC patterns from the given side-channel waveforms.
  This routine is used both for the initial extraction of the IaPAM and for the
  pre-attack extraction.

  Args:
    - waves: side-channel waveforms; numpy array of shape (nWaves, nSamples).
    - patternIMAC: pattern for the important MACs; numpy array of shape (nSamples,).
    - patternNIMACExec: pattern for the non-important executed MACs; numpy array
      of shape (nSamples,).
    - patternNIMACExec: pattern for the non-important skipped MACs; numpy array
      of shape (nSamples,).

  Return: the extracted IaPAM; numpy array of shape (imgWidth * imgHeight * numNeurons,).
  """

  if waves.size == 0:
    print("No waves to analyse. Exiting.")
    return

  # Array containing the IaPAM identified for each waveform.
  IaPAMs = np.zeros(shape = (waves.shape[0], p.imgWidth * p.imgHeight * p.nNeurons), dtype = bool)
  orderExecMACs = np.zeros(shape = (waves.shape[0], p.imgWidth * p.imgHeight * p.nNeurons), dtype = np.uint8)
  execMACs = np.zeros(shape = (waves.shape[0], p.imgWidth * p.imgHeight * p.nNeurons), dtype = bool)
  # Arrays containing only the identified executed MACs (important and not) for each waveform.
  # To better distinguish each MAC, we pad them with 'padImac' additional samples set
  # to '0'.
  IMACs = np.zeros(shape = (waves.shape[0], waves.shape[1]), dtype = np.float32)
  NIMACExecs = np.zeros(shape = (waves.shape[0], waves.shape[1]), dtype = np.float32)

  #fig = plt.figure(figsize = (7.5, 2), dpi = 300)
  #gs = fig.add_gridspec(1, hspace = 0)
  #axs = gs.subplots(sharex = True, sharey = True)

  with alive_bar(waves.shape[0]) as bar:
    for index, w in enumerate(waves):
      corrlIMAC = np.asarray( [c[0] for c in enumerate(pattern_detection.correlation(w, patternIMAC)) if c[1] > 0.92] )
      corrlNIMACExec = np.asarray( [c[0] for c in enumerate(pattern_detection.correlation(w, patternNIMACExec)) if c[1] > 0.92] )
      corrlNIMACSkip = np.asarray( [c[0] for c in enumerate(pattern_detection.correlation(w, patternNIMACSkip)) if c[1] > 0.92] )

      if corrlIMAC.shape[0] == 0:
        print(f"Skipping {index}/{waves.shape[0]}: no IMAC identified.")
        continue
      if corrlNIMACExec.shape[0] == 0:
        print(f"Skipping {index}/{waves.shape[0]}: no NIMAC (Executed) identified.")
        continue
      if corrlNIMACSkip.shape[0] == 0:
        print(f"Skipping {index}/{waves.shape[0]}: no IMAC (Skipped) identified.")
        continue
 
      # These "masks" are the original side-channel waveform, but only the
      # samples with the identified patterns are kept.
      IMACMask = np.zeros(shape = w.shape)
      NIMACMaskExec = np.zeros(shape = w.shape)
      NIMACMaskSkip = np.zeros(shape = w.shape)

      pos = 0
      # Used to store the NIMAC skipped not overlapping the NIMAC executed.
      updatedCorrlNIMACSkip = []
      for i in corrlIMAC:
        # If we reached the end of the waveform, stop extraction for this
        # waveform.
        if (w.shape[0] < i + len(patternIMAC)):
          break
        if (IMACs.shape[1] - pos < len(patternIMAC)):
          break

        IMACMask[i:i + len(patternIMAC)] = w[i:i + len(patternIMAC)]
        IMACs[index, pos:pos + len(patternIMAC)] = w[i:i + len(patternIMAC)]
        #axs[index].text(i - (len(patternIMAC) * 1.5), -0.4, r'$\textit{I}$')
        pos = pos + len(patternIMAC)
      pos = 0
      for i in corrlNIMACExec:
        NIMACMaskExec[i:i + len(patternNIMACExec)] = w[i:i + len(patternNIMACExec)]
        NIMACExecs[index, pos:pos + len(patternNIMACExec)] = w[i:i + len(patternNIMACExec)]
        #axs[index].text(i - (len(patternNIMACExec) * 1.25), -0.4, r'$\textit{E}$')
        pos = pos + len(patternNIMACExec)
      for i in corrlNIMACSkip:
        # Filter out the overlapping patterns (assuming skipped NIMACs exhibit a shorter pattern)
        # FIXME: This check should be done according to the correlation value
        #if not (NIMACMaskExec[i:i + len(patternNIMACSkip)] != 0.0).any():
        NIMACMaskSkip[i:i + len(patternNIMACSkip)] = w[i:i + len(patternNIMACSkip)]
        #axs[index].text(i - len(patternNIMACSkip) * 2, -0.4, r'$\textit{S}$')
        updatedCorrlNIMACSkip.append(i)

      updatedCorrlNIMACSkip = np.array(updatedCorrlNIMACSkip, dtype = np.uint32)

      #axs.set_xlabel(r"\textbf{Sample}")
      #axs.set_ylabel(r"\textbf{Power}")
      #axs.ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0), useMathText=True)
      #axs.plot(w[160:3820], linewidth = 0.5, label=r"\textbf{Raw}")
      #axs.plot(IMACMask[160:3820] + 0.5, alpha = 0.75, linewidth = 0.5, label = r"\textbf{IMAC}")
      #axs.plot(NIMACMaskExec[160:3820] + 0.75, alpha = 0.75, linewidth = 0.5, label = r"\textbf{NIMAC (Exec)}")
      #axs.plot(NIMACMaskSkip[160:3820] + 1.00, alpha = 0.75, linewidth = 0.5, label = r"\textbf{NIMAC (Skip)}")
      #plt.show(block = True)

      IaPAM = []
      lastNumNIMACs = 0
      # build the IaPAM:
      # For each identified IMAC pattern:
      # 1. Compute how many preceeding NIMAC (executed and skipped) pattern are
      # present (numNIMACs)
      # 2. Append a number #numNIMACs of zeros
      # 3. Append a 1 (the identified IMAC)
      for i in corrlIMAC:
        numNIMACs = len(np.where((corrlNIMACExec < i))[0]) + len(np.where((updatedCorrlNIMACSkip < i))[0])
        #print(f"[{i}: {lastNumNIMACs}; {numNIMACs})")
        IaPAM = IaPAM + [0] * (numNIMACs - lastNumNIMACs) + [1]
        lastNumNIMACs = numNIMACs
      # If there are not more IMACs, and we have not explored the whole image,
      # append a number of 0 equivalent to the remaining image's pixels.
      pad = p.imgWidth * p.imgHeight * p.nNeurons - len(IaPAM)
      IaPAMs[index] = IaPAM + [0] * pad

      waveExecMACs = []
      lastNumNIMACs = 0
      # build list of processed pixels:
      # For each waveform:
      #   For each identified IMAC and executed NIMAC
      #   1. Compute how many preceeding skipped NIMACs are present
      #   2. Append a number of #numNIMACs of zeros
      #   3. Append a 1 (the identified IMAC or executed NIMAC)
      corrlExecMACs = np.sort(np.concatenate((corrlIMAC, corrlNIMACExec)))
      for i in corrlExecMACs:
        numNIMACs = len(np.where((updatedCorrlNIMACSkip < i))[0])
        waveExecMACs = waveExecMACs + [0] * (numNIMACs - lastNumNIMACs) + [1]
        lastNumNIMACs = numNIMACs
      pad = p.imgWidth * p.imgHeight * p.nNeurons - len(waveExecMACs)
      execMACs[index] = waveExecMACs + [0] * pad

      # 0 -> Skipped; 1 -> Executed; 2 -> Important
      orderExecMACs[index] = execMACs[index].astype(np.uint8) + IaPAMs[index].astype(np.uint8)
      bar()

  #box = axs[1].get_position()
  #axs[1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
  #leg = axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
  #        fancybox=True, shadow=False, ncol=5)
  #for line in leg.get_lines():
  #  line.set_linewidth(1.75)

  #plt.tight_layout()
  #plt.savefig('waveforms-macs-sequence-extraction.svg')
  #plt.show()

  execMACs = np.asarray([ u.bytify(u.reverse(e)) for e in execMACs ], dtype = np.uint8)

  # Consistency check.
  # We first check what is the most recurrent IaPAM.
  # If the analysed IaPAM has inconsistency rate of less than 0.25, we keep it and end the extraction.
  # Otherwise, we explore the next IaPAM.
  # In case of an incosistent IaPAM, the check identifies also the corresponding waveform for later
  # processing (e.g., discarding it from future analyses).
  for i in range(0, IaPAMs.shape[0]):
    consistencies, inconsistencies = checkIaPAMs(IaPAMs, i)

    if not inconsistencies.size:
      saveIaPAM(u.bytify(u.reverse(IaPAMs[i])), IMACs, NIMACExecs, orderExecMACs, execMACs, suffix)
      return u.reverse(IaPAMs[i])

    inconsistencyRate = inconsistencies.size / (inconsistencies.size + consistencies.size)
    print(f">> Detected the following inconsistencies (rate = {inconsistencyRate}):")
    print(f">> Waveform indeces: {inconsistencies}")

    if inconsistencyRate > 0.25:
      continue

    print(">> Less than 0.25 inconsistencies")
    saveIaPAM(u.bytify(u.reverse(IaPAMs[consistencies[0]])), IMACs, NIMACExecs, orderExecMACs, execMACs, suffix)

    print(f">> Save inconsistencies in {datapathCirc}/inconsistencies.npy")
    np.save(f"{datapathCirc}/inconsistencies-index.npy", inconsistencies)
    np.save(f"{datapathCirc}/inconsistencies.npy", u.bytify(u.reverse(IaPAMs[inconsistencies])))
    return u.reverse(IaPAMs[consistencies[0]])

  print(">> Detected too many inconsistencies")


def main():

  imac = np.load('../artefacts/patterns/pattern-IMAC.npy')
  nimacexec = np.load('../artefacts/patterns/pattern-NIMACExec.npy')
  nimacskip = np.load('../artefacts/patterns/pattern-NIMACSkip.npy')

  for db in databases_protected:
    w = np.load(f'{datapathProt}/waveforms-extract-{db}.npy')
    extractIaPAM(w, imac, nimacexec, nimacskip, suffix = db)

if __name__ == '__main__':
  main()
