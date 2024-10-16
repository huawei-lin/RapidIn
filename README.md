# RapidIn
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/huawei-lin/LLMsEasyFinetune/blob/master/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

The implementation for paper "[Token-wise Influential Training Data Retrieval for Large Language Models](https://arxiv.org/abs/2405.11724)" (Accepted at ACL 2024)

We are still working on it, and adding more new features to this repo.

## Quick Start
Clone this repo to your local device.
```
git clone https://github.com/huawei-lin/RapidIn.git
cd RapidIn
```

Create a new environment by [anaconda](https://www.anaconda.com/download) and install the dependencies.
```
conda create -n RapidIn python=3.10
conda activate RapidIn
pip install -r requirements.txt
```

We provide an example of config file in `./examples` dir. You can modify it based on your requirements.

Once you have a config file, you can run:
```
python MP_main.py --config='config.json'
```

## Running Example
We also provide an example in `./examples`.

In `./examples`, we provide a dataset `alpaca_52k_with_5k_howdy_backdoor.jsonl`, which is a combination of the alpaca 52k dataset and 5k "Howdy!" backdoored data samples as described in our paper. We also provide a test dataset with 3 data samples in `backdoor_test.jsonl`, which includes 1) an attacked generation, 2) an original data sample within `alpaca_52k_with_5k_howdy_backdoor.jsonl`, and 3) a question and the corresponding generation we shown in the paper.

As shown in the `./example/config_caching.json`, we used [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b) as the base model, and use [huaweilin/rapidin-alpaca-llama2-7b](https://huggingface.co/huaweilin/rapidin-alpaca-llama2-7b) as the LoRA adapter.

### Caching Stage
You have to run the caching stage first. It will generate a output dir and a gradients caching dir. Please make sure that you have at least 8GB of free space.
```
cd ./example
python ../MP_main.py --config='./config_caching.json'
```

### Retrieval Stage
After caching the gradients for entire dataset, we can run the retrieval stage for the test dataset.
```
python ../MP_main.py --config='./config_retrieval.json'
```

### Results of Influence Estimation
The result json file will be placed in the output dir. We are still working on the script for visualization. We will release the code of this part by 6/24/2024.

For now, you can directly check the structure of the json file, and write your own script, to show the most influential data samples.

## Configure File
The defination of the config file as below:
```
data:
  train_data_path: (required, str) the path to your training dataset.
  test_data_path: (required, str) the path to your test generation.

influence:
  outdir: (required, str) the path to program standard output.
  seed: (optional, int, default: 42) the random seed.
  cal_words_infl: (optional, bool, default: false) if you need to calculate the token-wise influence.
  grads_path: (optional, str) the path to save the full gradient vectors or RapidGrads.
  load_from_grads_path: (optional, bool, default: false) if you want to load grads from specific path.
  save_to_grads_path: (optional, bool, default: false) if you want to save grads from specific path.
  n_threads: (optional, int, default: 1) the number of threads for each GPU.
  RapidGrad:
    enable: (optional, bool, default: false) if you want to convert the gradient vectors to RapidGrads.
    RapidGrad_K: (optional, int, default: 65536) expected dimensionality.
    shuffle_lambda: (optional, int, default: 20) the number of shuffles.
  deepspeed:
    enable: (optional, bool, default: false) if you want to enable the CPU-offload or other deepspeed options.
    config_path: (optional, str, default: None) the path to deepspeed config.
  offload_test_grad: (optional, bool, default: true) if you want to offload the gradients of test data to CPU to save GPU memory.
  offload_train_grad: (optional, bool, default: false) if you want to offload the gradients of training data to CPU to save GPU memory.
  top_k: (optional, int, default: 1000) output top-# influential data.

model:
  model_path: (required, str) the path to model.
  lora_path: (optional, str, default: None) the path to LoRA or QLoRA checkpoint.
  max_length: (optional, int, default: 512) the max length of the model.
  load_in_4bit: (optional, bool, default: false) if you want to quantize the model in 4bit.
```

## Citation
```
@inproceedings{rapidin,
  author       = {Huawei Lin and
                  Jikai Long and
                  Zhaozhuo Xu and
                  Weijie Zhao},
  title        = {Token-wise Influential Training Data Retrieval for Large Language Models},
  booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational
                  Linguistics, {ACL}},
  address      = {Bangkok, Thailand},
  year         = {2024},
}
```


