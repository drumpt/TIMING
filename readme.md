# TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation

[![arXiv](https://img.shields.io/badge/arXiv-2506.05035-b31b1b.svg)](https://arxiv.org/abs/2506.05035)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15671176.svg)](https://doi.org/10.5281/zenodo.15671176)
![](https://github.com/drumpt/drumpt.github.io/blob/main/content/publications/timing/featured.png)



[**TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation**](https://arxiv.org/abs/2506.05035)<br>
Hyeongwon Jang*, Changhun Kim*, Eunho Yang (*: equal contribution)<br>
International Conference on Machine Learning (**ICML**), 2025 (Spotlight Presentation, 313/12107=2.6%)



## Introduction
Official implementation for **TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation**. TIMING is implemented in PyTorch and tested on different time series datasets, including switch-feature, state, Mimic-III, PAM, Epilespy, boiler, freezer, and wafer. Our overall experiments are based on [time_interpret](https://github.com/josephenguehard/time_interpret), [ContraLSP](https://github.com/zichuan-liu/ContraLSP), [TimeX++](https://github.com/zichuan-liu/TimeXplusplus), [WinIT](https://github.com/layer6ai-labs/WinIT). 
Sincere thanks to each of the original authors!



## Installation instructions

```shell script
conda create -n timing python==3.10.16
conda activate timing
pip install -r requirement.txt --no-deps
```
The requirements.txt file is used to install the necessary packages into a virtual environment.

To test with switch-feature, additional setup is required.

```shell script
git clone https://github.com/TimeSynth/TimeSynth.git
cd TimeSynth
python setup.py install
cd ..
python synthetic/switchstate/switchgenerator.py
```



## Reproducing experiments

We have divided our experiments into two categories: Synthetic and Real.

All experiments can be executed using scripts located in scripts/real, scripts/hmm, or scripts/switchfeature.

This is an example execution for MIMIC-III (ours)
```shell script
bash scripts/real/train.sh
bash scripts/real/run_mimic_our.sh
bash scripts/real/run_mimic_baseline.sh
```

Due to differences in our training environments, our results reported in the paper may not be fully reproducible using the provided training scripts alone.

If you require access to the exact model checkpoints used in our experiments, please contact the authors directly. (janghw0911@kaist.ac.kr)

All results will be stored in the current working directory.

And then save parsing results:
```shell script
python real/parse.py --model state --data mimic3 --top_value 100
python real/parse.py --model state --data mimic3 --experiment_name baseline --top_value 100
```

All parsed results will be saved in the results/ directory.
