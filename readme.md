# Timing: Temporality-Aware Integrated Gradients for Time Series Explanation

Code for **Timing: Temporality-Aware Integrated Gradients for Time Series Explanation** submitted in ICML 2025. Timing is implemented in PyTorch and tested on different time series datasets, including switch-feature, state, Mimic-III, PAM, Epilespy, boiler, freezer, wafer. Our overall experiments are based on [time_interpret](https://github.com/josephenguehard/time_interpret), [ContraLSP](https://github.com/zichuan-liu/ContraLSP), [TimeX++](https://github.com/zichuan-liu/TimeXplusplus), [WinIT](https://github.com/layer6ai-labs/WinIT). 

Sincere thanks to each of the original authors!

## Installation instructions

```shell script
conda create -n timing python=3.10
conda activate timing
pip install -r requirement.txt
pip install -U tensorboardX
```
The requirements.txt file is used to install the necessary packages into a virtual environment.

To test with winit and fit, additional setup is required.

```shell script
git clone https://github.com/TimeSynth/TimeSynth.git
cd TimeSynth
python setup.py install
cd ..
```

## Reproducing experiments

We have divided our experiments into two categories: Synthetic and Real.

All experiments can be executed using scripts located in scripts/real, scripts/hmm, or scripts/switchfeature.

This is example execution for MIMIC-III (ours)
```shell script
bash scripts/real/train.sh # 

bash scripts/real/run_mimic_our.sh
bash scripts/real/run_mimic_baseline.sh
```

All results will be stored in the current working directory.

And then save parsing results:
```shell script
python real/parse.py --model state --data mimic3 --top_value 100
python real/parse.py --model state --data mimic3 --experiment_name baseline --top_value 100
```

All parsed results will be saved in the results/ directory.