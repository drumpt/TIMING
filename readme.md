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











## Further Reading
1, [**Transformers in Time Series: A Survey**](https://arxiv.org/abs/2202.07125), in IJCAI 2023.
[\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)

**Authors**: Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, Liang Sun

```bibtex
@inproceedings{wen2023transformers,
  title={Transformers in time series: A survey},
  author={Wen, Qingsong and Zhou, Tian and Zhang, Chaoli and Chen, Weiqi and Ma, Ziqing and Yan, Junchi and Sun, Liang},
  booktitle={International Joint Conference on Artificial Intelligence(IJCAI)},
  year={2023}
}
```

2, [**Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook**](https://arxiv.org/abs/2310.10196), in *arXiv* 2023.
[\[GitHub Repo\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

**Authors**: Ming Jin, Qingsong Wen*, Yuxuan Liang, Chaoli Zhang, Siqiao Xue, Xue Wang, James Zhang, Yi Wang, Haifeng Chen, Xiaoli Li (IEEE Fellow), Shirui Pan*, Vincent S. Tseng (IEEE Fellow), Yu Zheng (IEEE Fellow), Lei Chen (IEEE Fellow), Hui Xiong (IEEE Fellow)

```bibtex
@article{jin2023lm4ts,
  title={Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook}, 
  author={Ming Jin and Qingsong Wen and Yuxuan Liang and Chaoli Zhang and Siqiao Xue and Xue Wang and James Zhang and Yi Wang and Haifeng Chen and Xiaoli Li and Shirui Pan and Vincent S. Tseng and Yu Zheng and Lei Chen and Hui Xiong},
  journal={arXiv preprint arXiv:2310.10196},
  year={2023}
}
```

3, [**Position Paper: What Can Large Language Models Tell Us about Time Series Analysis**](https://arxiv.org/abs/2402.02713), in *arXiv* 2024.

**Authors**: Ming Jin, Yifan Zhang, Wei Chen, Kexin Zhang, Yuxuan Liang*, Bin Yang, Jindong Wang, Shirui Pan, Qingsong Wen*


```bibtex
@article{jin2024position,
   title={Position Paper: What Can Large Language Models Tell Us about Time Series Analysis}, 
   author={Ming Jin and Yifan Zhang and Wei Chen and Kexin Zhang and Yuxuan Liang and Bin Yang and Jindong Wang and Shirui Pan and Qingsong Wen},
  journal={arXiv preprint arXiv:2402.02713},
  year={2024}
}
```
4, [**AI for Time Series (AI4TS) Papers, Tutorials, and Surveys**](https://github.com/qingsongedu/awesome-AI-for-time-series-papers)
