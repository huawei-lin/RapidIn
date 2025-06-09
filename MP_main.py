import argparse
import RapidIn as rapidin
import torch.multiprocessing as mp
import random
import numpy as np

CONFIG_PATH = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=CONFIG_PATH, type=str)
    args = parser.parse_args()
    config_path = args.config_path

    rapidin.init_logging()
    config = rapidin.get_config(config_path)
    print(config)

    random.seed(int(config.influence.seed))
    np.random.seed(int(config.influence.seed))

    rapidin.calc_infl_mp(config)
    print("Finished")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # mp.set_start_method('forkserver')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()
