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

from alive_progress   import alive_bar
from os import listdir

def computeGE(ranking):
  """ Compute the guessing entropy for the given ranking.

  Args:
    - ranking: matrix (numExperiments, numSamples, numCandidates)

  Return:
    - ges: guessing entropy of each candidate per sample; matrix (numSamples, numCandidates)
  """

  ges = np.mean(np.log2(ranking), axis = 1)

  return ges

## This script computes the Guessing Entropy (GE) from several datasets of ranking.
## The script calculates the GE for a given range of weights and a range of
## neurons:
## - firstInput: specifies the first weight (included)
## - lastInput: specifies the last weight (excluded)
## - firstNeuron: specifies the first neuron (included)
## - lastNeuron: specifies the last neuron (excluded)
##
## The script is parameterised to consider three different implementations
## (listed by the variable 'available_implementations'):
## - unprotected
## - protected
## - circumvented
## The implementation to analyse is indicatated by the 'implementation'
## variable.
## The script handles also traces partitioned according to the
## execution/skipping of a certain non-important weight (listed by the variable
## 'available_filteredTypes').
## To analyse such trace dataset, use the variable 'filteredWaveforms'.
## Note that 'filteredWaveforms = True' will analyse ALL the partitioned trace
## datasets.
##
## The above variables are only used to select the corresponding dataset,
## assuming that the dataset path contains such strings.
## They do not change any other parameters in this script.
##
## By default, the script saves the computed GEs.
## The datapaths where the script peaks the rankings and saves the plots and GEs
## are defined by 'datapathRanking', 'datapathPlots', 'datapathGE'.
##
## The script provides a 'toPlot' boolean variable to save in SVG format the
## GE.
##
## NOTA BENE: certain plot-relatede parameteres (e.g., the plot title) are
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

datapath='../data'
datapathRanking=f'{datapath}/rankings'
datapathPlots=f'{datapath}/plots'
datapathGE=f'{datapath}/ges'

# Computation parameters
firstInput = 7
lastInput = 8
numInputs = 8

firstNeuron = 0
lastNeuron = 2
numNeurons = lastNeuron - firstNeuron

numWeights = 32

# MLP implementations considered.
available_implementations = ['unprotected', 'protected', 'circumvented']
# Partition traces according to what weight to skip
available_filteredTypes   = ['exec-5', 'non-exec-5', 'exec-7', 'non-exec-7']

# What implementation to consider
implementation = 'circumvented'
# Set to 'True' if you want to repeat the experiments in Section V.F.
filteredWaveforms = True

assert implementation in available_implementations

# Reverse the set of weights
weights = np.flip(np.split(p.weights, p.weights.shape[0] // 8), axis = 1).reshape(-1)

if toPlot:
  #f, axs = plt.subplots(numNeurons, 1, figsize = (3.15, 2), sharex = True, sharey = True, dpi = 300)
  fig = plt.figure(figsize = (7.15, 5), dpi = 300)
  plt.suptitle(r"\textbf{Protected}")
  fig.supylabel(r"\textbf{Guessing Entropy}")

  gs = fig.add_gridspec(numNeurons, hspace = 0)
  axs = gs.subplots(sharex = True, sharey = True)


## Compute the GE for each neuron and each weight.
for n in range(firstNeuron, lastNeuron):
  weightsSetBegin = numWeights * n
  weightsSetEnd = weightsSetBegin + numWeights

  # Create the vectors containing the GE.
  # Each vector contain the GE of the true weight of each considered weight.
  if filteredWaveforms:
    geExecPerInput = []
    geNonExecPerInput = []
  else:
    gePerInput = []

  with alive_bar(lastInput - firstInput) as bar:
    for i in range(firstInput, lastInput):
      # Select the true weight value for the currently analysed weight
      trueWeight = weights[weightsSetBegin:weightsSetEnd][i]

      if filteredWaveforms:

        # Collect the ranking for the partitioned traces
        rankingsExec = []
        rankingsNonExec = []

        pathExec = f"{datapathRanking}/{implementation}/neuron-{n}/input-{i}/*extract-exec*"
        pathNonExec = f"{datapathRanking}/{implementation}/neuron-{n}/input-{i}/*extract-non-exec*"

        pathsExec = g.glob(pathExec)
        pathsNonExec = g.glob(pathNonExec)

        if (pathsExec == [] or pathsNonExec == []):
          continue

        for p in pathsExec:
          rankingsExec.append(np.load(p))
        rankingsExec = np.asarray(rankingsExec, dtype = np.uint8)
        rankingsExec = rankingsExec.transpose(1, 0, 2, 3)[:, :, :]

        for p in pathsNonExec:
          rankingsNonExec.append(np.load(p))
        rankingsNonExec = np.asarray(rankingsNonExec, dtype = np.uint8)
        rankingsNonExec = rankingsNonExec.transpose(1, 0, 2, 3)[:, :, :]

        # Compute the GE for both trace partitions.
        # Store a matrix of dimension (numWaves, numSamples, numCandidates).
        gesExec = computeGE(rankingsExec)
        geExecTrueWeight = np.min(gesExec[:, :, trueWeight - 1], axis = 1)
        geExecPerInput.append(geExecTrueWeight)

        gesNonExec = computeGE(rankingsNonExec)
        geNonExecTrueWeight = np.min(gesNonExec[:, :, trueWeight - 1], axis = 1)
        geNonExecPerInput.append(geNonExecTrueWeight)

      else:
        # Collect the rankings.
        path = f"{datapathRanking}/{implementation}/neuron-{n}/input-{i}/*"
        paths = g.glob(path)

        rankings = []
        for p in paths:
          rankings.append(np.load(p))
        rankings = np.asarray(rankings, dtype = np.uint8)
        rankings = rankings.transpose(1, 0, 2, 3)[:, :, :]

        # Compute the GE.
        # Store a matrix of dimension (numWaves, numSamples, numCandidates).
        ges = computeGE(rankings)

        # Get the GE for the true weight value.
        geTrueWeight = np.min(ges[:, :, trueWeight - 1], axis = 1)
        gePerInput.append(geTrueWeight)
      bar()

  if toPlot:
    axs[n].ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0), useMathText=True)
    axs[n].grid(axis = 'both', color = "lightgrey", linewidth = '0.5', linestyle = 'dashed')
    axs[n].set_ylim(-0.99, 7.99)
    axs[n].text(0.0, 6.0, r'\textbf{Neuron} ' + f'${n}$')
    #axs[n].set_ylabel(r'\textbf{GE}')
    axs[n].set_xlabel(r'\textbf{Traces}')
    axs[n].xaxis.set_ticks(range(0, 251, 50))

  if filteredWaveforms:
    geExecPerInput = np.asarray(geExecPerInput)
    geNonExecPerInput = np.asarray(geNonExecPerInput)

    if toPlot:
      for i, ge in enumerate(geExecPerInput[:, :]):
        axs[n].plot(ge, label = r"$w_{5}$", linestyle = 'solid')
      for i, ge in enumerate(geNonExecPerInput[:, :]):
        axs[n].plot(ge, label = r"$w_{7}$", linestyle = 'dotted')


    np.save(f'{datapathGE}/{implementation}/neuron-{n}/{implementation}-ge-input-{i}-exec.npy', geExecPerInput)
    np.save(f'{datapathGE}/{implementation}/neuron-{n}/{implementation}-ge-input-{i}-non-exec.npy', geNonExecPerInput)
  else:
    gePerInput = np.asarray(gePerInput)
    if toPlot:
      for i, ge in enumerate(gePerInput[:, :]):
        axs[n].plot(ge, label = r"$w_{1}$" + f"${i + 1}$")
    np.save(f'{datapathGE}/{implementation}/neuron-{n}/{implementation}-ge-input-{i}.npy', gePerInput)

if toPlot:
  box = axs[numNeurons - 1].get_position()
  axs[numNeurons - 1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
  leg = axs[numNeurons - 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=4, columnspacing = 0.75)

  for line in leg.get_lines():
      line.set_linewidth(1.75)


  plt.tight_layout()
  if filteredWaveforms:
    plt.savefig(f'{datapathPlots}/{implementation}/{implementation}-ge-input-{firstInput}-{lastInput}-filtered.svg')
  else:
    plt.savefig(f'{datapathPlots}/{implementation}/{implementation}-ge-input-{firstInput}-{lastInput}.svg')
