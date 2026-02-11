# Training the ML model 
This is a brief description of how to train an ML model to be used by [automatic direction selection script](https://git.astron.nl/RD/LINC/-/blob/master/scripts/tune_demixing_parameters.py). The training should be run in this directory.

## Requirements
Hardware:
A GPU is recommended, but not essential, as the data generation and training is pretty light weight.

The following executables are required:

- Python libraries: PyTorch, numpy, scipy, astropy
- For creation of blank MS [makems](https://git.astron.nl/ro/lofar/-/blob/master/CEP/MS/src/makems.cc)
- For generating data [sagecal](https://github.com/nlesc-dirac/sagecal)

After building the above software, edit ```./calibration/generate_data.py``` to point to their correct locations in your installation. The configuration of the problem (like adding additional directions in the sky) should be done by editing ```demixingenv.py```.

# Step 1: generate training data

```
python makedata.py 
```

# Step 2: train a MSP (deep neural net with 3 layers) model

```
python train_regressor.py
```

That is it, you can copy the trained model ```test_regressor.model``` to your LINC installation.

# Optional step 3: train a TSK (fuzzy) enhanced ML model

```
python train_tsk.py
```

# Optional step 4: evaluate trained models
Re-run step 1 above to generate separate data for evaluation (not advisable to use the data that is used for training). Thereafter, run

```
python evaluate_tsk_msp.py
```

to compare the two trained ML models in steps 2 and 3.

wo 11 feb 2026 13:24:40 CET
