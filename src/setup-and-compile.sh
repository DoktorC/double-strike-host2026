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

# Covenient script to compile out of the box the Double-Strike proof of concept.

# List of directories to check/create
directories=(
  "objdir-CWLITEARM/TinyEngine/src/kernels/int_forward_op/"
  "objdir-CWLITEARM/TinyEngine/src/kernels/fp_requantize_op/"
  "objdir-CWLITEARM/TinyEngine/src/kernels/fp_backward_op/"
  "objdir-CWLITEARM/TinyEngine/codegen/Source/"
  "objdir-CWLITEARM/hal/stm32f3/CMSIS/nn/src/ConvolutionFunctions/"
)

# Loop through the directories
for dir in "${directories[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "Directory $dir does not exist. Creating..."
    mkdir -p "$dir"
  else
    echo "Directory $dir already exists."
  fi
done


# Compile that thing!
# Nota bene: it compiles the firmware with optimisation level 03.
make 
