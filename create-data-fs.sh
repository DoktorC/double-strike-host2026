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

#!/usr/bin/env bash

# Script setting up the 'data' folder.
# The 'data' folder contains the all the artefacts from the acquisition
# campaigns and analyses.

nInputs=7
nNeurons=4

implementations=("unprotected" "protected" "circumvented")
analyses=("corrls" "rankings" "ges" "plots")

for implementation in "${implementations[@]}"; do
  mkdir -p "./data/$implementation"
done

for analysis in "${analyses[@]}"; do
  for implementation in "${implementations[@]}"; do
    for n in $(seq 0 ${nNeurons}); do
      for i in $(seq 1 ${nInputs}); do
        mkdir -p "./data/${analysis}/${implementation}/neuron-${n}/input-${i}/"
      done
    done
  done
done
