import sys
import json
import logging
from pathlib import Path
from datetime import datetime as dt
import datetime
from tqdm import tqdm
import collections.abc
import torch

tqdm_dict = {}
def save_json(json_obj, json_path, append_if_exists=False,
              overwrite_if_exists=False, unique_fn_if_exists=True):
    """Saves a json file

    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f'{str(json_path.stem)}_{time}'\
                                               f'{str(json_path.suffix)}'

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, 'w+') as fout:
            json.dump(json_obj, fout, indent=None)
        return json_path

    if append_if_exists:
        if json_path.exists():
            with open(json_path, 'r') as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)
            with open(json_path, 'w+') as fout:
                json.dump(read_file, fout, indent=None)
            return json_path

    with open(json_path, 'w+') as fout:
        json.dump(json_obj, fout, indent=None)

    return json_path

def load_json(json_path, key2int=True):
    def covert_key_to_int(d):
        new_dict = {}
        for k, v in d.items():
            if k.isnumeric() == True:
                k = int(k)
            if isinstance(v, dict):
                v = covert_key_to_int(v)
            new_dict[k] = v
        return new_dict

    with open(json_path, 'r') as f:
        result = json.load(f)

    result = covert_key_to_int(result)
    return result



def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True, new_line=False, run_time=None, cur_time=None):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    if text not in tqdm_dict.keys():
        tqdm_dict[text] = tqdm(total=last_step, desc=text)
    tqdm_dict[text].n = current_step
    tqdm_dict[text].refresh()


def init_logging(filename=None):
    """Initialises log/stdout output

    Arguments:
        filename: str, a filename can be set to output the log information to
            a file instead of stdout"""
    log_lvl = logging.INFO
    log_format = '%(asctime)s: %(message)s'
    if filename:
        logging.basicConfig(handlers=[logging.FileHandler(filename),
                                      logging.StreamHandler(sys.stdout)],
                            level=log_lvl,
                            format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, level=log_lvl,
                            format=log_format)


def get_default_config():
    """Returns a default config file"""
    config = {
        "data": {
            "train_data_path": None,
            "test_data_path": None,
            "begin_id": None,
            "end_id": None,
            "test_begin_id": None,
            "test_end_id": None,
        },
        "influence": {
            "outdir": "outdir",
            "seed": 42,
            "IF": {
              "recursion_depth": 5,
              "r_averaging": 3,
              "scale": 50000,
             },
            "cal_words_infl": False,
            "grads_path": None,
            "load_from_grads_path": False,
            "save_to_grads_path": False,
            "delete_model": False,
            "n_threads": 1,
            "RapidGrad": {
                "enable": False,
                "RapidGrad_M": 1,
                "RapidGrad_K": 65536,
                "shuffle_lambda": 20,
                "multi_k_save_path_list": None # only assign by program
            },
            "deepspeed": {
                "enable": False,
                "config_path": None,
            },
            "offload_test_grad": True,
            "offload_train_grad": False,
            "calculate_infl_in_gpu": False,
            "skip_test": False,
            "skip_influence": False,
            "infl_method": "TracIn", # TracIn, IF. (default: TracIn)
            "top_k": 1000,
        },
        "model": {
            "model_path": None,
            "lora_path": None,
            "max_length": None,
            "load_in_4bit": False,
        }
    }

    return config

def sanity_check(config):
    if config.influence.RapidGrad.enable and isinstance(config.influence.RapidGrad.RapidGrad_K, list):
        if config.influence.skip_test == False \
                or config.influence.skip_influence == False:
            print("RapidGrad_K is a list, set `skip_test` and `skip_influence` to True")
            config.influence.skip_test = True
            config.influence.skip_influence = True
        if config.influence.save_to_grads_path == False or config.influence.grads_path is None:
            assert("RapidGrad_K is a list, set `save_to_grads_path` to True and assign `grads_path`.")


class Struct:
    """The recursive class for building and representing objects with."""
    def __init__(self, obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))


def get_config(config_path):
    """Returns a  config file"""
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    config = get_default_config()
    config = update(config, json.load(open(config_path)))
    config = Struct(config)
    sanity_check(config)
    return config
