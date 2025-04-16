This code trains a fuzzy controller using reinforcement learning, the fuzzy controller is trained to provide the optimal directions for direction dependent calibration (demixing).

How to run the script 

```
 main_sac.py  --use_hint --iteration 10000 --seed 222
```

Using different random *--seed*, we can train an ensemble of models. The trained ensemble model can be used to determine the final parameters to use for the fuzzy controller.


## Requirements
Hardware:
A GPU is recommended

Executables:
sagecal,makems,excon

Python:
pytorch, numpy, scipy, python-casacore, gymnasium, scikit-fuzzy

## Notes
- Before running the training, copy LOFAR LBA/HBA antenna tables to this directory, as LBA and HBA.
- Also edit *../calibration/generate_data.py* to point to makems, excon, sagecal_gpu, and sagecal-mpi_gpu depending on your installation.

wo 16 apr 2025  9:11:57 CEST
