import os
from typing import Dict, Optional, Sequence
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
import json
import copy
import torch
import logging
from torch.utils.data import Dataset
from peft import (
    PeftModel,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import random
import transformers
from safetensors.torch import load_file
from datasets import load_from_disk


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
NEW_CACHE_DIR = "new_cache_dir/"
prompt_no_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def create_special_tokens_dict(tokenizer):
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    return special_tokens_dict


def get_model_tokenizer(config, **kwargs):
    tokenizer = get_tokenizer(config, **kwargs)
    model = get_model(config, tokenizer, **kwargs)
    return model, tokenizer


def get_model(config, tokenizer=None, **kwargs):
    device_map = kwargs.get("device_map", None)
    model_path = config.model_path
    logging.warning("Loading model...")
    model = None
    bnb_config = None
    if config.load_in_4bit == True:
        print("load_in_4bit:", config.load_in_4bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # load_in_8bit=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    if device_map is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=False, cache_dir=NEW_CACHE_DIR
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            cache_dir=NEW_CACHE_DIR,
        )

    if tokenizer is not None:
        special_tokens_dict = create_special_tokens_dict(tokenizer)

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

    if config.load_in_4bit == True:
        model = prepare_model_for_kbit_training(model)

    # lora will be added to the base model
    if config.checkpoint is not None:
        model = PeftModel.from_pretrained(
            model, config.checkpoint, is_trainable=True, cache_dir=NEW_CACHE_DIR
        )

        # load lora from checkpoint
        checkpoint_safetensors = os.path.join(
            config.checkpoint, "adapter_model.safetensors"
        )  # only LoRA model - LoRA config above has to fit
        adapters_weights = load_file(checkpoint_safetensors)
        set_peft_model_state_dict(model, adapters_weights)
    elif config.lora_path is not None:
        # load lora from lora path in huggingface
        logging.warning(f"Loading lora adapter...")
        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(
            model,
            config.lora_path,
            is_trainable=True,
            device_map=device_map,
            cache_dir=NEW_CACHE_DIR,
        )
        model.print_trainable_parameters()

    model.config.use_cache = False
    model.is_parallelizable = True
    model.model_parallel = True

    #     if torch.__version__ >= "2":
    #         model = torch.compile(model)
    # model.eval()
    model.train()
    return model


def get_tokenizer(config, **kwargs):
    model_path = config.model_path
    logging.warning("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=NEW_CACHE_DIR)
    tokenizer.max_length = config.max_length
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})
    return tokenizer


def get_dataset_size(data_path):
    content = None
    with open(data_path) as f:
        content = f.readlines()
    return len(content)


def read_data(data_path, type="supervised"):
    list_data_dict = None
    if type == "supervised":
        with open(data_path) as f:
            list_data_dict = [json.loads(line) for line in f]
    elif type == "unsupervised":
        # TODO: Consier use the tokenizer to transfer id back to text.
        dataset = load_from_disk(data_path)
        list_data_dict = dataset["input_ids"]
    else:
        raise ValueError("Unsupported data type: ", type)
    return list_data_dict


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def supervised_data_preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess the supervised data with sources and targets.

    Params:
        sources: input prompts
        targets: expected model responses
        tokenizer:

    Returns:
        A dict with 3 keys:
            - input_ids: tokenized ids of the examples (an example is a source + a target)
            - labels: same as input_id, but with masks on [:source_len - 1]
            - input_ids_lens: source input ids lens
    """
    examples = [s + t for s, t in zip(sources, targets)]  # exapmles is source + targets
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[: source_len - 1] = IGNORE_INDEX
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=sources_tokenized["input_ids_lens"],
    )


class TrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        type: str = "supervised",
        shuffle: bool = True,
        shuffle_seed: int = 42,
        load_idx_list=None,
        begin_id=None,
        end_id=None,
    ):
        super(TrainDataset, self).__init__()
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.begin_id = begin_id
        self.end_id = end_id

        data_dict = None
        if type == "supervised":
            data_dict = self.construct_supervised_data_dict(
                tokenizer, data_path, load_idx_list
            )
        elif type == "unsupervised":
            data_dict = self.construct_unsupervised_data_dict(tokenizer, data_path)
        else:
            raise ValueError("Unsupported data type: ", type)

        logging.warning("Done tokenizing inputs...")

        if load_idx_list is None:
            load_idx_list = list(range(len(data_dict["input_ids"])))

        s = list(range(len(load_idx_list)))
        if self.shuffle == True:
            random.seed(self.shuffle_seed)
            random.shuffle(s)

        self.input_ids = [data_dict["input_ids"][i] for i in s]
        self.sorted_index = [load_idx_list[i] for i in s]
        # self.list_data_dict = [ list_data_dict[i] for i in s ]
        self.labels = [data_dict["labels"][i] for i in s]
        self.input_ids_lens = [data_dict["input_ids_lens"][i] for i in s]

    def construct_supervised_data_dict(self, tokenizer, data_path, load_idx_list):
        logging.warning("Loading data...")
        list_data_dict = read_data(data_path)
        logging.warning("Formatting supervised inputs...")
        sources = [prompt_no_input.format_map(example) for example in list_data_dict]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]
        if self.begin_id is not None:
            if self.end_id is None:
                self.end_id = len(sources)
            load_idx_list = list(range(self.begin_id, self.end_id))
        if load_idx_list is not None:
            sources = [sources[x] for x in load_idx_list]
            targets = [targets[x] for x in load_idx_list]
        print(f"sources: {len(sources)}, targets: {len(targets)}")

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = supervised_data_preprocess(sources, targets, tokenizer)
        print("===data dict keys", data_dict.keys())
        print("--", len(data_dict["input_ids"]))
        return data_dict

    # use the tokenized and grouped data
    def construct_unsupervised_data_dict(self, tokenizer, data_path):
        dataset = load_from_disk(data_path)
        input_ids = [torch.tensor(id) for id in dataset["input_ids"]][self.begin_id:self.end_id]
        labels = [torch.tensor(id) for id in dataset["labels"]][self.begin_id:self.end_id]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=[
                torch.tensor(len(input_id)) for input_id in dataset["input_ids"]
            ],
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return (
            self.input_ids[i],
            self.labels[i],
            self.input_ids_lens[i],
            self.sorted_index[i],
        )


class TestDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        type: str = "supervised",
    ):
        super(TestDataset, self).__init__()
        self.list_data_dict = []
        self.input_ids = []
        self.input_ids_lens = []
        self.hotwords = []

        if data_path is None or len(data_path) == 0:
            return

        if type == "supervised":
            self.construct_supervised_test_data(data_path, tokenizer)
        elif type == "unsupervised":
            self.construct_unsupervised_test_data(data_path)
        else:
            raise ValueError("Unsupported data type: ", type)

    def construct_supervised_test_data(self, data_path, tokenizer):
        logging.warning("Loading data...")
        list_data_dict = []
        if data_path is not None and len(data_path) != 0:
            list_data_dict = read_data(data_path)

        logging.warning("Formatting inputs...")
        sources = [prompt_no_input.format_map(example) for example in list_data_dict]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]
        hotwords = [
            [hw.strip() for hw in example.get("hotwords", "").split("|") if hw != ""]
            for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = supervised_data_preprocess(sources, targets, tokenizer)

        print(f"Detected hotwords: {hotwords}")
        self.labels = []
        for hotwords_list, label_tokens in zip(hotwords, data_dict["labels"]):
            if len(hotwords_list) == 0:
                self.labels.append(label_tokens)
                continue
            label_tokens = label_tokens.tolist()
            hotwords_tokens = [
                tokenizer.encode(x, add_special_tokens=False) for x in hotwords_list
            ]
            new_label = [-100 for _ in range(len(label_tokens))]
            label_tokens_len = len(label_tokens)
            for hotword in hotwords_tokens:  # hotword: [1, 2, 3]
                hotword_len = len(hotword)
                for i in range(label_tokens_len):
                    if i + hotword_len >= label_tokens_len:
                        break
                    if hotword == label_tokens[i : i + hotword_len]:
                        new_label[i : i + hotword_len] = label_tokens[
                            i : i + hotword_len
                        ]
            self.labels.append(torch.LongTensor(new_label))

        self.list_data_dict = list_data_dict
        self.input_ids = data_dict["input_ids"]
        self.input_ids_lens = data_dict["input_ids_lens"]
        self.hotwords = hotwords

    # Use the processed (tokenized and grouped) data
    def construct_unsupervised_test_data(self, data_path):
        dataset = load_from_disk(data_path)
        input_ids = [torch.tensor(id) for id in dataset["input_ids"]]
        labels = [torch.tensor(id) for id in dataset["labels"]]

        self.input_ids = input_ids
        self.labels = labels
        self.input_ids_lens = [
            torch.tensor(len(input_id)) for input_id in dataset["input_ids"]
        ]
        print("===Test data ready: ", len(self.input_ids))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.input_ids[i], self.labels[i], self.input_ids_lens[i]
