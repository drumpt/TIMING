# TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation
[![arXiv](https://img.shields.io/badge/arXiv-2506.05035-b31b1b.svg)](https://arxiv.org/abs/2506.05035)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15671176.svg)](https://doi.org/10.5281/zenodo.15671176)

[**TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation**](https://arxiv.org/abs/2506.05035)<br>
Hyeongwon Jang*, Changhun Kim*, Eunho Yang (*: equal contribution)<br>
International Conference on Machine Learning (**ICML**), 2025 (Spotlight Presentation, 313/12107=2.6%)

![](https://github.com/drumpt/drumpt.github.io/blob/main/content/publications/timing/featured.png)



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

Due to differences between our training environment and the released code, the paper’s results may not be fully reproducible with the training scripts alone. All XAI evaluations were conducted on a single NVIDIA RTX 3090 GPU (24GB VRAM). We have publicly released [all model checkpoints](https://drive.google.com/file/d/1nX9x2iBTcFUykHsKAj_1xQftWvZtc1B6/view?usp=sharing), except those trained on the restricted-access MIMIC-III dataset.

Researchers who need MIMIC-III checkpoints may contact the authors directly. (janghw0911@kaist.ac.kr)

All results will be stored in the current working directory.

And then save parsing results:
```shell script
python real/parse.py --model state --data mimic3 --top_value 100
python real/parse.py --model state --data mimic3 --experiment_name baseline --top_value 100
```

All parsed results will be saved in the results/ directory.

## Citation
```
@inproceedings{jang2025timing,
  title={{TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation}},
  author={Jang, Hyeongwon and Kim, Changhun and Yang, Eunho},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

