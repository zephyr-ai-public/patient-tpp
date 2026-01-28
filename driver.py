import argparse

from patient_tpp.easy_tpp_zai.config_factory.config import ZaiConfig
from patient_tpp.easy_tpp_zai.runner.tpp_runner import ZaiTPPRunner


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/config.yaml",
        help="Configuration yaml",
    )

    parser.add_argument(
        "--experiment_id",
        type=str,
        required=False,
        default="AttNHP_zai_train",
        help="Experiment id in the config file",
    )

    args = parser.parse_args()

    config = ZaiConfig.build_from_yaml_file(
        args.config, experiment_id=args.experiment_id
    )

    model_runner = ZaiTPPRunner.build_from_config(config)

    model_runner.run()


if __name__ == "__main__":
    main()
