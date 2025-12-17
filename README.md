# Double Strike: Breaking Approximation-Based Side-Channel Countermeasures for DNNs

This repository contains the software (analysis code, MLP implementation, TinyEngine commit) employed for
the research paper *Double Strike: Breaking Approximation-Based Side-Channel Countermeasures for DNNs*.

Analysis Artefacts
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17965887.svg)](https://doi.org/10.5281/zenodo.17965887)

Analysis Code Archive
[![DOI](https://zenodo.org/badge/1087268667.svg)](https://zenodo.org/badge/latestdoi/1087268667)

# Repository Structure

The repository follows this structure:

```
 *
 `-- artefacts
     `-- firmware
     `-- patterns
     `-- acqusitions-info
 `-- python-utils
 `-- src
     `-- hal
     `-- simpleserial
     `-- TinyEngine
     `   `-- codegen
     `   `   `-- Source
     `   `   `-- Include
     `   `-- src
     `       `-- kernels
     `           `-- int_forward_op
```

 * `python-utils` contains the python scripts employed for the communication with the board and analysis code;
 * `src` contains the implementations to run on the ChipWhisperer-Lite;
 * `hal` contains a modified version of the HAL bundled with the Chipwhisperer software;
 * `TinyEngine` contains the TinyEngine framework and the C implementation of the MACPruned MLP model;
    * `codegen/Include/genModel.h` contains the parameters (e.g., weights and biases) of the MLP;
    * `codegen/Source/genModel.c` defines the layers of the MLP and their invocation order;
    * `src/kernels/int_forward_op` contains the definition of each layer; for our MLP, we have added MACPruning to the `convolve_1x1_s8_oddch.c`. 
 * `simpleserial` contains the library to communicate with the SimpleSerial protocol;
 * `artefacts` contains the information on the acquisition campaigns, the firmware, and the side-channel patterns used in the *Double Strike* article.

## Required Software and Libraries

For the compilation of the MLP implementation:

 * `arm-none-eabi-gcc` compilation toolchain (version `15:10.3-2021.07-4`).

For the python-based tools:

| Package             | alive-progress   | chipwhisperer   | importlib_metadata   | matplotlib   | numpy   | pip      | scared |
|:-------------------:|:----------------:|:---------------:|:--------------------:|:------------:|:-------:|:--------:|--------|
| Version             | 3.2.0            | 5.7.0           | 8.5.0                | 3.9.4        | 1.26.4  | 24.3.1   | 1.1.9  |

 
A `requirements.txt` lists these modules.

We suggest the creation of a virtual environment before installing the required modules:

```
  pyenv install 3.9.5
  pyenv virtualenv 3.9.5 double-strike
  pyenv local double-strike

  pip3 install -r requirements.txt
```

Concerning the Chipwhisperer-Lite target, we used the `0.64.0` firmware.

## Communication Between Host Computer and ChipWhisperer-Lite®

The comunication uses the SimpleSerial protocol, version 2.1 and baud rate of `115200`.

## Setting up the `data` Folder

Before running any acquisition campaign and analysis, you must create the `data`
folder.
Unless you modify the datapaths in the scripts contained in
`python-utils`, this folder is the destination of all the acquisition campaigns
and analyses.

To setup this folder, run the script `create-data-fs.sh`.

## Life Cycle of the Experimental Campaign

The experimental campaign runs in four phases:

  1. Compile the firmware
  2. Flash the firmware on the ChipwhisperLite®
  3. Collect power traces
  4. Circumvent MACPruning

To ease Steps 2 and 3, we provide you a `capture-cwlite.py` script, which provides a simple REPL interface to accomplish these steps.
You can also use the standard python interpreter, but we invite you to read the `main.c` file and `test-vector.py` module as we do not cover it here.

Before continuing, a little introduction to the `test-vector.py`

#### The Interface for Flashing and Collecting Traces

The `capture-cwlite.py` provides a simple REPL interface to flash and capture the power traces from the Chipwhisperer-Lite®.
We designed the interface to minimise the burden implied by setting the connection with the target, flash it and recover the power traces.

When ran, the interface opens the connection with the target and sets communication parameters to default settings; the `baud_rate` is an exception, which is set to `115200` to provide low-latency data transfer.

The interface accepts the following commands:

| Command     | c                 | d                  | e                 | h                   | l                       | f                | q    |
|:-----------:|:-----------------:|:------------------:|:-----------------:|:-------------------:|:-----------------------:|:----------------:|:----:|
| Description | Collect waveforms | Disable MACPruning | Enable MACPruning | Show splash message | Load test vector driver | Flash the target | Quit |


You can find two commands to enable/disable the MACPruning countermeasure (default to `disable`), and one command to reload the `test-vector.py` module without quitting and running again the `capture-cwlite.py` interface.
It may be useful if, for any reason, you have to change the `test-vector.py` module.

You can interrupt any running command with `CTRL+C`, which brings you back to the command selection.
Any unhandled exception does not exit the REPL interface, but brings you back to the command selection.

To quit the REPL, you have to explictly ask it with the `q` command.
The communication with the target is automatically closed.

### Compile the Firmware

The compilation process compiles the DNN model (in `TinyEngine`), the HAL library and the SimpleSerial protocol in a unique binary file.
This project uses a modified version of Chipwisperer's Makefile, adapted to integrate the compilation of the DNN model.
By default, the build system sets the optimisation level to `-O3`.

For convenience, a bash script takes care of creating the necessary directories and run the compilation:

```
./setup-and-compile.sh
```
Eventually, you should get the `main-CWLITEARM.{hex, bin, elf}` files.

The folder `artefacts/firmware` contains the sha256 of the firmware compiled with debug information and without it. 

### Flash the Firmware on the Chipwhisperer-Lite®

To flash the Chipwhisperer-Lite® via the REPL interface, simply use the command `f`.
It will flash the firmware located by the internal variable `fwpath` (default to `./main-CWLITEARM.hex`).

### Collect Power Traces

To collect the power traces via the REPL interface, simply use the command `c`.

The `test-vector.py` is a python script for the collection of the power traces from the Chipwhisperer-Lite®.
It automatically handles the generation of the:
  - IaPAM (generated once and used during the acquisition of the side-channel traces);
  - toExecTable (generated each time before acquiring a new side-channel trace);
  - 32-bit Inputs (generated each time before acquiring a new side-channel trace).

The script randomly generates the three features.
To guarantee the reuse of the same IaPAM and toExecTables with different inputs, the generation happens through distinct RNG instances, each seeded with a distinct seed: IaPAM and toExecTable use the same instance, whereas the inputs use a different one.

You can change these seeds with the variable `seedInputs` and `seedMACPruning` (`capture-cwlite.py`).

The REPL automatically saves each trace collection in a directory pointed by the variable `datapath`.
Each trace dataset starts with the prefix `waveforms-` followed by the time at which the collection started.

Furthermore the REPL automatically saves the generated `IaPAM`, the `toExecTable`s, the inputs and further information characterising the trace collection:

|      platform        |      scope                             |             target                 |        waves           |        seeds           |     MACPruning      |
|:--------------------:|:--------------------------------------:|:----------------------------------:|:----------------------:|:----------------------:|:-------------------:|
|       CWLITE      | fw_version, gain, adc, clock, trigger  |                baud, chip          |     nwaves, nsamples   |        inputs, MACPruning          |   enabled/disable   |


As for the trace datasets, each file starts with a prefix identifying it (e.g., toExecTables -> `toExecTables`) followed by the time at which the collection started.

### Circumvent MACPruning

The circumvention of the MACPruning countermeasure relies on a preprocessing of the collected power traces.
Such preprocessing produces new traces preserving only the samples related to the execution of important MACs.
You can find further details in our research paper.

The preprocessing happens in four steps:

  1. Patterns Identification
  2. MACs Classification (`macs_classification.py`) -- Indentify and extract the position and execution order of each MAC in the given traces.
  3. Filter Processed Pixels (`build_hyps.py`) -- Build the leakage hypotheses considering only the executed MACs.
  4. Important MACs Concatenation (`mac_classification.py`) -- Create new power traces from the concatenation of the important MACs.

Concering the first step -- Patterns Identification -- there is no python module dedicated to it; you have to manually identify the patters in a sample trace (or build your own module :-)).

Concerning step 4, this is transparently carried out by the `macs_classification.py` module.

## Further Analyses

The `python-utils` folder contains other two modules:

* `compute_mean_corrl.py` computes the mean correlation score over a certain number of trace datasets;
* `partition_circum_waveforms.py` partition a dataset of traces in two sets: one captured during the execution of a certain MAC, one captured while a certain MAC is not executed.

The first module refers to the correlation analysis we carried out in Section V.C (*Considering Extended Traces*).

The second module refers to the analysis we carried out in Section V.F (*Unintended Recovery of Non-important Weights*).
 
*Nota Bene:* if you are using relatives path in the scripts, execute them from the `python-utils` folder.

## How to Reproduce the Experiments

Once collected the side-channel traces, recovered the IaPAM and the non-important MACs, and computed the leakage hypotheses, you can use the `compute_ranking.py` and `compute_ge.py` script to reproduce each experiment reported in our article.

The python module `compute_ranking.py` computes the ranking of each weigth candidate value from the given power traces and leakage hypotheses.
To rank, the module uses the Pearson's Correlation score.
The module contains several parameters to tune the analysis (e.g., number of traces to analyse, what weights target, what portion of the traces).

To compute the Guessing Entropy from the weights candidates ranking, you can rely on the module `compute_ge.py`.

For more information, we invite you to read the documentation inside each module.

### Reproducibility conditions

To reproduce the same results from the paper, you should:
  * use the same firmware and side-channel patterns;
  * Use the same IaPAM;
  * use the same sequence of executed/skipped non-important MACs.

We provide the firmware and side-channel patterns in the `artefacts` repository.

For reproducing the same IaPAM and sequence of executed/skipped non-important MACs, you must use the same numpy version reported in this repository, and the same seeds we provide in the json files describing the acquisition campaign parameters.

We provide in an external archive the:

  * collected side-channel traces;
  * the inputs fed to the DNN implementations;
  * the sequence of random values to determine whether skip or not a non-important weight;
  * the sequence of executed MACs/processed weights;
  * the parameters charactersing the side-channel acquisition campaign.

You can find this archive at the following URL: https://zenodo.org/records/17965887.

# Cite in your Documents

If you make use of any part of this software (derivated works too), please consider to cite our original work and this repository as follows:

```
@inproceedings{CasalRPS26,
  author={Lorenzo, Casalino and Maria Mendez, R\'{e}al, and Jean-Christophe, Pr\'{e}votet and Rub\'{e}n, Salvador},
  booktitle={2026 IEEE International Symposium on Hardware Oriented Security and Trust (HOST)},
  title={Double Strike: Breaking Approximation-Based Side-Channel Countermeasures for DNNs},
  year={2026}
}
```

Thank you :-)!
