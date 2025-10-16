#!/usr/bin/env python3

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

import glob               as g
import matplotlib.pyplot  as plt
import matplotlib.style   as style
import numpy              as np
import params             as p

from os import listdir

def computeMeanCorrl(corrls):
  """ Compute the mean Pearson's correlation score from the given set of
  Pearson's correlation scores.

  Args:
    - corrls: matrix (numNeurons, numExperiments, numSamples, numCandidates)

  Return:
    - avgCorrls: the average correlation score of each candidate per sample; matrix (numSamples, numCandidates)
  """

  avgCorrls = np.mean(corrls, axis = 1)

  return avgCorrls


## This script computes the mean Pearson's correlation score (MCS) from several
## datasets of correlation scores.
## The script calculates the MCS for a given range of weights and a range of
## neurons:
## - firstInput: specifies the first weight (included)
## - lastInput: specifies the last weight (excluded)
## - firstNeuron: specifies the first neuron (included)
## - lastNeuron: specifies the last neuron (excluded)
##
## The script is designed to work only on 'unprotected' implementation.
##
## The above variables are only used to select the corresponding dataset,
## assuming that the dataset path contains such strings.
## They do not change any other parameters in this script.
##
## By default, the script saves the computed MCS.
## The datapaths where the script peaks the rankings and saves the plots and MCS
## are defined by 'datapathCorrls' and 'datapathPlots'.
##
## The script provides a 'toPlot' boolean variable to save in SVG format the
## MCS.
##
## NOTA BENE: certain plot-related parameteres (e.g., the plot title) are
## hardcoded.

# Matplotlib parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\boldmath'

style.use('tableau-colorblind10')

toPlot = True

datapath = f'../data/'
datapathCorrls = f'{datapath}/corrls'
datapathPlots = f'{datapath}/plots'

# Computation parameters
firstInput = 1
lastInput = 8
numInputs = lastInput - firstInput

firstNeuron = 3
lastNeuron = 4
numNeurons = lastNeuron - firstNeuron

numWeights = 32

implementation = 'unprotected'

# Reverse the set of weights
weights = np.flip(np.split(p.weights, p.weights.shape[0] // 8), axis = 1).reshape(-1)

if toPlot:
  #f, axs = plt.subplots(numNeurons, 1, figsize = (7.5, 2), sharex = True, sharey = True, dpi = 300)
  #fig = plt.figure(figsize = (3.5, 5), dpi = 300)
  fig = plt.figure(figsize = (7.1, 5.5), dpi = 300)
  plt.suptitle(r"\textbf{Unprotected}")
  fig.supylabel(r"\textbf{Correlation Score}")

  gs = fig.add_gridspec(numInputs, hspace = 0)
  axs = gs.subplots(sharex = True, sharey = False)

for n in range(firstNeuron, lastNeuron):
  weightsSetBegin = numWeights * n
  weightsSetEnd = weightsSetBegin + numWeights

  for i in range(firstInput, lastInput):
    path = f"{datapathCorrls}/{implementation}/neuron-{n}/input-{i}/*"
    accumPaths = g.glob(path)
    accumCorrls = []

    if accumPaths == []:
      continue

    for ap in accumPaths:
      accumCorrls.append(np.load(ap))

    accumCorrls = np.asarray(accumCorrls, dtype = np.float32)
    #accumCorrls = accumCorrls.transpose(1, 0, 2, 3)

    # Matrix of dimension (numWaves, numSamples, numCandidates)
    avgAccumCorrls = computeMeanCorrl(accumCorrls)

    maxCorrlScore = np.max(avgAccumCorrls)
    trueWeight = weights[weightsSetBegin:weightsSetEnd][i] 

    np.save(f'{datapathCorrls}/{implementation}/neuron-{n}/{implementation}-avg-corrl-input-{i}.npy', avgAccumCorrls)

    if toPlot:
      axs[i - 1].ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0), useMathText=True)
      axs[i - 1].grid(axis = 'both', color = "lightgrey", linewidth = '0.5', linestyle = 'dashed')
      axs[i - 1].text(940, maxCorrlScore / 2, r'\textbf{Weight} ' + fr'${i}$', size = 8)
      axs[i - 1].set_xlabel(r'\textbf{Sample}')
      axs[i - 1].plot(np.max(np.delete(avgAccumCorrls, trueWeight - 1, axis = 2), axis = 2)[0, :1000], color = '#898989', label = r'\textbf{Best Weight Value}', linewidth = 1)
      axs[i - 1].plot(avgAccumCorrls[0, :1000, trueWeight - 1], label = r'\textbf{True Weight Value}', linewidth = 1)
      axs[i - 1].margins(y = 0.20)
      axs[i - 1].set_yticks([0, np.float32(f'{maxCorrlScore}'[:4])])

if toPlot:
  box = axs[numInputs - 1].get_position()
  axs[numInputs - 1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
  leg = axs[numInputs - 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=4, columnspacing = 0.75)
  
  for line in leg.get_lines():
      line.set_linewidth(1.75)
  
  
  #labels = [ float(f'{x}'[:3]) for x in np.arange(0, 1.1, 0.1) ]
  #plt.xticks(ticks = range(0, 1100, 100), labels = [ fr'${x}$' for x in labels ])
  #plt.xticks(ticks = range(0, 1100, 100), labels = np.arange(0, 1.1, 0.1))
  plt.tight_layout()
  plt.savefig(f'{datapathPlots}/{implementation}/{implementation}-corrl-input-{firstInput}-{lastInput}-neuron-{firstNeuron}.svg')
  plt.show()
