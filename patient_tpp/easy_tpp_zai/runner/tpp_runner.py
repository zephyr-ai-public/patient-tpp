import difflib
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Generator, Optional, Union

import numpy as np
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.runner.base_runner import Runner
from easy_tpp.utils import (
    MetricsHelper,
    Registrable,
    RunnerPhase,
    Timer,
    concat_element,
    count_model_params,
    get_stage,
    get_unique_id,
    logger,
    save_pickle,
    set_seed,
)
from easy_tpp.utils.const import Backend, PredOutputIndex

from patient_tpp.easy_tpp_zai.config_factory.runner_config import ZaiRunnerConfig
from patient_tpp.easy_tpp_zai.model.torch_model.torch_attnhp import ZaiNumericAttNHP
from patient_tpp.easy_tpp_zai.preprocess.data_loader import ZaiTPPDataLoader
from patient_tpp.easy_tpp_zai.preprocess.numeric_data_loader import (
    ZaiTPPNumericDataLoader,
)
from patient_tpp.easy_tpp_zai.torch_wrapper import ZaiTorchModelWrapper
from patient_tpp.easy_tpp_zai.utils.metrics_tracker import ZaiMetricsTracker


@Runner.register(name="zai_tpp")
class ZaiTPPRunner(Runner, Registrable):
    """TPP runner suitable for use with invariant, indicative, and numerical features."""

    def __init__(
        self,
        runner_config: ZaiRunnerConfig,
        unique_model_dir: bool = False,
        **kwargs: Any,
    ) -> None:
        # super(ZaiTPPRunner, self).__init__(runner_config, unique_model_dir, **kwargs)

        self.runner_config = runner_config
        # re-assign the model_dir
        if unique_model_dir:
            runner_config.model_dir = (
                runner_config.base_config.specs["saved_model_dir"]
                + "_"
                + get_unique_id()
            )

        self.save_log()

        self.metrics_tracker = ZaiMetricsTracker()
        if self.runner_config.trainer_config.metrics is not None:
            self.metric_functions = self.runner_config.get_metric_functions()

        self._init_model()

        pretrain_dir = self.runner_config.model_config.pretrained_model_dir
        if pretrain_dir is not None:
            self._load_model(pretrain_dir)

        data_config = runner_config.data_config
        backend = runner_config.base_config.backend
        kwargs = runner_config.trainer_config.get_yaml_config()
        self._data_loader = ZaiTPPDataLoader(
            data_config=data_config, backend=backend, **kwargs
        )
        num_numeric_event_types = (
            42
            if data_config.data_specs.num_numeric_event_types is None
            else int(data_config.data_specs.num_numeric_event_types)
        )
        numeric_pad_token_id = (
            41
            if data_config.data_specs.numeric_pad_token_id is None
            else int(data_config.data_specs.numeric_pad_token_id)
        )
        if data_config.data_specs.use_numeric_features:
            self._numeric_data_loader = ZaiTPPNumericDataLoader(
                data_config=data_config,
                backend=backend,
                numeric_bins=self.runner_config.model_config.numeric_bins,
                num_numeric_event_types=num_numeric_event_types,
                pad_token_id=numeric_pad_token_id,
                **kwargs,
            )
        self.timer = Timer()

    def _init_model(self) -> None:
        """Initialize the model."""
        self.use_torch = self.runner_config.base_config.backend == Backend.Torch

        if self.use_torch:
            set_seed(self.runner_config.trainer_config.seed)

            self.model = TorchBaseModel.generate_model_from_config(
                model_config=self.runner_config.model_config
            )
            self.numeric_model = ZaiNumericAttNHP(
                model_config=self.runner_config.model_config,
                num_event_types_pad=self.runner_config.data_config.data_specs.num_numeric_event_types
                * self.runner_config.model_config.numeric_bins
                + 3,
                use_pad_token_id=self.runner_config.data_config.data_specs.numeric_pad_token_id,
                num_tracks=self.runner_config.data_config.data_specs.num_numeric_event_types,
            )
            self.model_wrapper = ZaiTorchModelWrapper(
                self.model,
                self.numeric_model,
                self.runner_config.base_config,
                self.runner_config.model_config,
                self.runner_config.trainer_config,
            )
            numer_params = count_model_params(self.model)

        else:
            raise (NotImplementedError("Tensorflow not supported yet."))

        info_msg = f"Num of model parameters {numer_params}"
        logger.info(info_msg)

    def train(
        self,
        train_loader: Optional[ZaiTPPDataLoader] = None,
        valid_loader: Optional[ZaiTPPDataLoader] = None,
        test_loader: Optional[ZaiTPPDataLoader] = None,
        **kwargs: Any,
    ) -> TorchBaseModel:
        """Train the model.

        Args:
            train_loader (easy_tpp_zai.DataLoader, optional): data loader for train set. Defaults to None.
            valid_loader (easy_tpp_zai.DataLoader, optional): data loader for valid set. Defaults to None.
            test_loader (easy_tpp_zai.DataLoader, optional): data loader for test set. Defaults to None.

        Returns:
            model: _description_
        """
        # no train and valid loader from outside
        if train_loader is None:
            usable_train_loader = self._data_loader.train_loader(**kwargs)
        else:
            usable_train_loader = train_loader

        if valid_loader is None:
            usable_valid_loader = self._data_loader.valid_loader(**kwargs)
        else:
            usable_valid_loader = valid_loader

        # no test loader from outside and there indeed exists test data in config
        if test_loader is None and self.runner_config.data_config.test_dir is not None:
            usable_test_loader = self._data_loader.test_loader()
        else:
            usable_test_loader = test_loader

        logger.info(f"Data '{self.runner_config.base_config.dataset_id}' loaded...")

        numeric_train_loader = None
        numeric_valid_loader = None
        numeric_test_loader = None
        if hasattr(self, "_numeric_data_loader"):
            numeric_train_loader = self._numeric_data_loader.train_loader()
            numeric_valid_loader = self._numeric_data_loader.valid_loader()
            if self.runner_config.data_config.test_dir is not None:
                numeric_test_loader = self._numeric_data_loader.test_loader()

        timer = self.timer
        timer.start()
        model_id = self.runner_config.base_config.model_id
        logger.info(f"Start {model_id} training...")
        model = self._train_model(
            usable_train_loader,
            usable_valid_loader,
            test_loader=usable_test_loader,
            numeric_train_loader=numeric_train_loader,
            numeric_valid_loader=numeric_valid_loader,
            numeric_test_loader=numeric_test_loader,
            **kwargs,
        )
        logger.info(f"End {model_id} train! Cost time: {timer.end()}")
        return model

    def _train_model(
        self,
        train_loader: ZaiTPPDataLoader,
        valid_loader: ZaiTPPDataLoader,
        **kwargs: Any,
    ) -> TorchBaseModel:
        """Train the model.

        Args:
            train_loader (ZaiTPPDataLoader): data loader for the train set.
            valid_loader (ZaiTPPDataLoader): data loader for the valid set.
        """
        test_loader = kwargs.get("test_loader")
        secondary_loaders = {
            "numeric_train_loader": kwargs.get("numeric_train_loader"),
            "numeric_valid_loader": kwargs.get("numeric_valid_loader"),
            "numeric_test_loader": kwargs.get("numeric_test_loader"),
        }

        for i in range(self.runner_config.trainer_config.max_epoch):
            train_metrics = self.run_one_epoch(
                train_loader,
                RunnerPhase.TRAIN,
                secondary_loaders={
                    x: secondary_loaders[x] for x in ("numeric_train_loader",)
                },
            )

            message = (
                f"[ Epoch {i} (train) ]: train "
                + MetricsHelper.metrics_dict_to_str(train_metrics)
            )
            logger.info(message)

            self.model_wrapper.write_summary(i, train_metrics, RunnerPhase.TRAIN)

            # evaluate model
            if i % self.runner_config.trainer_config.valid_freq == 0:
                valid_metrics = self.run_one_epoch(
                    valid_loader,
                    RunnerPhase.VALIDATE,
                    secondary_loaders={
                        x: secondary_loaders[x] for x in ("numeric_valid_loader",)
                    },
                )
                self.model_wrapper.write_summary(i, valid_metrics, RunnerPhase.VALIDATE)

                message = (
                    f"[ Epoch {i} (valid) ]:  valid "
                    + MetricsHelper.metrics_dict_to_str(valid_metrics)
                )
                logger.info(message)

                updated = self.metrics_tracker.update_best(
                    "loglike", valid_metrics["loglike"], i
                )
                message_valid = f"Current best loglike on validation set is \
{self.metrics_tracker.current_best['loglike']:.4f} \
(updated at epoch {self.metrics_tracker.episode_best['loglike']})"
                logger.info(message_valid)
                if updated:
                    save_fn = (
                        f"{self.runner_config.base_config.specs['saved_model_dir']}.{i}"
                    )
                    self.model_wrapper.save(save_fn)
                    message_valid = (
                        f"Best updated at this epoch, model saved to {save_fn}."
                    )
                    logger.critical(message_valid)

                for m in ("rmse", "acc", "diff_ratio"):
                    updated = self.metrics_tracker.update_best(m, valid_metrics[m], i)
                    message_valid = f"current best {m} on validation set is {self.metrics_tracker.current_best[m]:.4f} \
(updated on epoch {self.metrics_tracker.episode_best[m]})"
                    logger.info(message_valid)

                if test_loader is not None:
                    test_metrics = self.run_one_epoch(
                        test_loader,
                        RunnerPhase.VALIDATE,
                        secondary_loaders={
                            x: secondary_loaders[x] for x in ("numeric_test_loader",)
                        },
                    )

                    message = (
                        f"[ Epoch {i} (test) ]: test "
                        + MetricsHelper.metrics_dict_to_str(test_metrics)
                    )
                    logger.info(message)

        self.model_wrapper.close_summary()
        return self.model

    def _save_model(self, model_dir: str) -> None:
        """Save the model.

        Args:
            model_dir (str): the dir for model to save.
        """
        if model_dir is None:
            model_dir = self.runner_config.base_config.specs["saved_model_dir"]
        self.model_wrapper.save(model_dir)
        logger.critical(f"Save model to {model_dir}")

    def _load_model(self, model_dir: str) -> None:
        """Load the model from the dir.

        Args:
            model_dir (str): the dir for model to load.
        """
        self.model_wrapper.restore(model_dir)
        logger.critical(f"Load model from {model_dir}")

    def _evaluate_model(self, data_loader: ZaiTPPDataLoader) -> dict[str, Any]:
        """Evaluate the model on the valid dataset.

        Args:
            data_loader (EasyTPP.DataLoader): data loader for the valid set

        Returns:
            dict: metrics dict.
        """

        eval_metrics = self.run_one_epoch(data_loader, RunnerPhase.VALIDATE)

        self.model_wrapper.write_summary(0, eval_metrics, RunnerPhase.VALIDATE)

        self.model_wrapper.close_summary()

        message = "Evaluation result: " + MetricsHelper.metrics_dict_to_str(
            eval_metrics
        )

        logger.critical(message)

        return eval_metrics

    def _gen_model(self, data_loader: ZaiTPPDataLoader) -> None:
        """Generation of the TPP, one-step and multi-step are both supported."""

        test_result = self.run_one_epoch(data_loader, RunnerPhase.PREDICT)
        i = 0
        slug = f"preds.{datetime.now().strftime('%Y%m%d')}.{i}.pkl"
        fn = os.path.join(self.runner_config.base_config.specs["eval_dir"], slug)
        while os.path.exists(fn):
            i += 1
            slug = f"preds.{datetime.now().strftime('%Y%m%d')}.{i}.pkl"
        save_pickle(fn, test_result)
        logger.critical(
            f"Predictions saved to {fn} ({len(test_result['pred'][0])} records)."
        )

    def run_one_epoch(
        self,
        _data_loader: ZaiTPPDataLoader,
        phase: RunnerPhase,
        secondary_loaders: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run one complete epoch.

        Args:
            data_loader: data loader object defined in model runner
            phase: enum, [train, dev, test]

        Returns:
            a dict of metrics
        """
        total_loss = 0
        total_number_events = 0
        epoch_label = []
        epoch_pred = []
        epoch_mask = []
        pad_index = self.runner_config.data_config.data_specs.pad_token_id
        return_intensities = False
        if secondary_loaders is None:
            secondary_loaders = {}

        metrics_dict: dict[str, Any] = OrderedDict()

        # Reset loader
        _data_loader = ZaiTPPDataLoader(
            data_config=self.runner_config.data_config,
            backend=self.runner_config.base_config.backend,
            **self.runner_config.trainer_config.get_yaml_config(),
        ).get_loader(
            {
                RunnerPhase.TRAIN: "train",
                RunnerPhase.PREDICT: "test",
                RunnerPhase.VALIDATE: "dev",
            }[phase]
        )

        _numeric_data_loader = None
        if self.runner_config.data_config.data_specs.use_numeric_features:
            _numeric_data_loader = ZaiTPPNumericDataLoader(
                data_config=self.runner_config.data_config,
                backend=self.runner_config.base_config.backend,
                **self.runner_config.trainer_config.get_yaml_config(),
            ).get_loader(
                {
                    RunnerPhase.TRAIN: "numeric_train",
                    RunnerPhase.PREDICT: "numeric_test",
                    RunnerPhase.VALIDATE: "numeric_dev",
                }[phase]
            )

        def generate_none() -> Generator[None, None, None]:
            while True:
                yield None

        batch_count = 0

        if phase in [RunnerPhase.TRAIN, RunnerPhase.VALIDATE]:
            batch_args = [
                iter(x) if x else generate_none()
                for x in (_data_loader, _numeric_data_loader)
            ]
            for batch, numeric_batch in zip(*batch_args):
                try:
                    batch_count += 1

                    (
                        batch_loss,
                        batch_number_events,
                        batch_pred,
                        batch_label,
                        batch_mask,
                    ) = self.model_wrapper.run_batch(batch, numeric_batch, phase=phase)
                    total_loss += batch_loss
                    total_number_events += batch_number_events
                    epoch_pred.append(batch_pred)
                    epoch_label.append(batch_label)
                    epoch_mask.append(batch_mask)
                    print(
                        f"Batch {batch_count}: loss {batch_loss:.2f} on {batch_number_events} events.\r",
                        end="",
                    )
                except StopIteration:
                    print(f"Stopping iteration at batch {batch_count}.", flush=True)
                    break

            avg_loss = total_loss / total_number_events

            metrics_dict.update(
                {"loglike": -avg_loss, "number_events": total_number_events}
            )

        else:
            return_intensities = self.runner_config.model_config.return_intensities
            batch_args = [
                iter(x) if x else generate_none()
                for x in (_data_loader, _numeric_data_loader)
            ]
            for batch, numeric_batch in zip(*batch_args):
                try:
                    batch_count += 1

                    batch_pred, batch_label = self.model_wrapper.run_batch(
                        batch,
                        numeric_batch,
                        phase=phase,
                        return_intensities=return_intensities,
                    )
                    epoch_pred.append(batch_pred)
                    epoch_label.append(batch_label)
                    print(f"Batch {batch_count}\r", end="")
                except StopIteration:
                    print(
                        f"Stopping prediction/eval iteration at batch {batch_count}.",
                        flush=True,
                    )
                    break

        # we need to improve the code here
        # classify batch_output to list
        pred_exists, label_exists = False, False
        epoch_pred_rv: Union[np.ndarray, tuple[np.ndarray, np.ndarray]] = np.array([])
        epoch_mask_rv = np.array([]).astype("bool")
        epoch_label_rv = np.array([])
        # Structure of epoch_foo is batch x (foo_dt, foo_types) x ~num_step_gen
        if epoch_pred[0][0] is not None:
            if return_intensities:
                # Partial entries (entries with fewer columns than the requested
                # num_step_gen for lookback. Filter these out until we find where
                # they're coming from.
                dts = [
                    x[0]
                    for x in epoch_pred
                    if len(x[0][0])
                    == self.runner_config.model_config.thinning.num_step_gen
                ]
                intensities = [
                    x[1]
                    for x in epoch_pred
                    if len(x[1][0])
                    == self.runner_config.model_config.thinning.num_step_gen
                ]
                epoch_pred_rv = (np.vstack(dts), np.vstack(intensities))
            else:
                epoch_pred_rv = concat_element(epoch_pred, pad_index)
                pred_exists = True
        if len(epoch_label) > 0 and epoch_label[0][0] is not None:
            epoch_label_rv = concat_element(epoch_label, pad_index)
            label_exists = True
            if len(epoch_mask) > 0:
                epoch_mask_rv = concat_element(epoch_mask, False)[
                    0
                ]  # retrieve the first element of concat array
                epoch_mask_rv = epoch_mask_rv.astype(bool)

        if (
            pred_exists
            and label_exists
            and (len(epoch_mask_rv) > 0)
            and not return_intensities
        ):
            metrics_dict.update(
                self.metric_functions(
                    epoch_pred_rv, epoch_label_rv, seq_mask=epoch_mask_rv
                )
            )

        if phase == RunnerPhase.PREDICT:
            metrics_dict.update({"pred": epoch_pred_rv, "label": epoch_label_rv})

        return metrics_dict

    def run(self, **kwargs: Any) -> Any:
        """Start the runner.

        Args:
            **kwargs (dict): optional params.

        Returns:
            EasyTPP.BaseModel, dict: the results of the process.
        """
        current_stage = get_stage(self.runner_config.base_config.stage)
        if current_stage == RunnerPhase.TRAIN:
            return self.train(**kwargs)
        elif current_stage == RunnerPhase.VALIDATE:
            return self.evaluate(**kwargs)
        else:
            return self.gen(**kwargs)


@MetricsHelper.register(
    name="diff_ratio", direction=MetricsHelper.MAXIMIZE, overwrite=False
)
def diff_ratio_metric_function(
    predictions: np.ndarray, labels: np.ndarray, **kwargs: Any
) -> float:
    """Compute difflib ratio metrics of the type predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: difflib's ratio score for the type predictions.
    """
    seq_mask = kwargs.get("seq_mask")
    if seq_mask is None:
        seq_mask = np.ones_like(predictions)
    masked_pred = np.where(seq_mask, predictions[PredOutputIndex.TypePredIndex], np.nan)
    masked_label = np.where(seq_mask, labels[PredOutputIndex.TypePredIndex], np.nan)
    ratios = []
    for pred, lab in zip(masked_pred, masked_label):
        matcher = difflib.SequenceMatcher(lambda x: np.isnan(x), pred, lab)
        ratios.append(matcher.ratio())
    ratio = float(np.mean(ratios))
    return ratio
