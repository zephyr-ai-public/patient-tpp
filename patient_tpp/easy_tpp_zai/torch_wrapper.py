"""Initialize a Pytorch model wrapper that feed into Model Runner"""

import os
from typing import Any, Optional

import torch
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.preprocess.event_tokenizer import BatchEncoding
from easy_tpp.torch_wrapper import TorchModelWrapper
from easy_tpp.utils import RunnerPhase, set_optimizer
from torch import Tensor

from patient_tpp.easy_tpp_zai.config_factory.config import ZaiConfig
from patient_tpp.easy_tpp_zai.config_factory.model_config import ZaiModelConfig, ZaiTrainerConfig


class ZaiTorchModelWrapper(TorchModelWrapper):
    def __init__(
        self,
        model: TorchBaseModel,
        numeric_model: TorchBaseModel,
        base_config: ZaiConfig,
        model_config: ZaiModelConfig,
        trainer_config: ZaiTrainerConfig,
    ):
        """A wrapper class for Torch backends, augmented to handle
        invariant and other non-core features.

        Args:
            model (BaseModel): a TPP model.
            numeric_model (BaseModel): a TPP model for numeric features.
            base_config (EasyTPP.Config): basic configs.
            model_config (easy_tpp_zai.ZaiModelConfig): model spec configs.
            trainer_config (easy_tpp_zai.ZaiTrainerConfig): trainer spec configs.
        """
        super(ZaiTorchModelWrapper, self).__init__(
            model, base_config, model_config, trainer_config
        )
        self.numeric_model = numeric_model
        self.numeric_model.to(self.device)

        if self.model_config.is_training:
            # set up optimizer
            optimizer = self.trainer_config.optimizer
            self.learning_rate = self.trainer_config.learning_rate
            self.numeric_opt = set_optimizer(
                optimizer, self.model.parameters(), self.learning_rate
            )

    def run_batch(
        self,
        batch: BatchEncoding,
        numeric_batch: BatchEncoding,
        phase: RunnerPhase,
        return_intensities: bool = False,
    ) -> Any:
        """Run one batch.

        Args:
            batch (EasyTPP.BatchEncoding): preprocessed batch data that go into the model.
            numeric_batch: data from the numeric features stream.
            phase (RunnerPhase): a const that defines the stage of model runner.

        Returns:
            tuple: for training and validation we return loss, prediction and labels;
            for prediction we return prediction.
        """
        numeric_layer: Optional[Tensor] = None
        batch = [batch.get(x, []) for x in batch.to(self.device)]
        if numeric_batch:
            numeric_batch = [numeric_batch[x] for x in numeric_batch.to(self.device)]

        if phase in (RunnerPhase.TRAIN, RunnerPhase.VALIDATE):
            # set mode to train
            is_training = phase == RunnerPhase.TRAIN
            self.model.train(is_training)
            self.numeric_model.train(is_training)

            # FullyRNN needs grad event in validation stage
            grad_flag = is_training if self.model_id != "FullyNN" else True
            # run model
            with torch.set_grad_enabled(grad_flag):
                numeric_loss, num_numeric_event, numeric_layer = (
                    self.numeric_model.loglike_loss(numeric_batch, numeric_layer=None)
                )
                loss, num_event, _ = self.model.loglike_loss(
                    batch, numeric_layer=numeric_layer
                )

            # Assume we don't do prediction on train set
            pred_dtime, pred_type, label_dtime, label_type, mask = (
                None,
                None,
                None,
                None,
                None,
            )

            # update grad
            if is_training:
                self.opt.zero_grad()
                if num_numeric_event > 0:
                    (numeric_loss / num_numeric_event).backward(retain_graph=True)
                (loss / num_event).backward()
                self.opt.step()
            elif self.model.event_sampler:
                self.model.eval()
                self.numeric_model.eval()
                with torch.no_grad():
                    # Omit the first item in the sequences ([:, 1:])
                    if batch[1] is not None and batch[2] is not None:
                        label_dtime, label_type = (
                            batch[1][:, 1:].cpu().numpy(),
                            batch[2][:, 1:].cpu().numpy(),
                        )
                    if batch[3] is not None:
                        mask = batch[3][:, 1:].cpu().numpy()
                    pred_dtime, pred_type = self.model.predict_one_step_at_every_event(
                        batch=batch, numeric_layer=numeric_layer
                    )
                    pred_dtime = pred_dtime.detach().cpu().numpy()
                    pred_type = pred_type.detach().cpu().numpy()
                    # label_dtime, label_type = thin_predictions_or_labels(label_dtime, label_type, pad_to_len=127)

            return (
                loss.item(),
                num_event + num_numeric_event,
                (pred_dtime, pred_type),
                (label_dtime, label_type),
                (mask,),
            )
        else:
            (
                time_seq_label,
                time_delta_seq_label,
                event_seq_label,
                batch_non_pad_mask_label,
                _,
                _,
                invars,
                _,
            ) = batch
            if numeric_batch is not None:
                (
                    num_time_seqs,
                    num_time_delta_seqs,
                    num_type_seqs,
                    _,
                    num_attention_mask,
                    _,
                    num_invars,
                    _,
                ) = numeric_batch

                self.numeric_model.eval()
                numeric_layer = self.numeric_model.forward(
                    num_time_seqs,
                    num_type_seqs,
                    num_attention_mask,
                    num_invars,
                    numeric_layer=None,
                    sample_times=num_time_seqs,
                ).to(num_time_seqs.device)

            pred_dtime, pred_type, label_dtime, label_type = (
                self.model.predict_multi_step_since_last_event(
                    batch=batch,
                    numeric_layer=numeric_layer,
                    return_intensities=return_intensities,
                )
            )
            pred_dtime = pred_dtime.detach().cpu().numpy()
            pred_type = pred_type.detach().cpu().numpy()
            label_dtime = label_dtime.detach().cpu().numpy()
            label_type = label_type.detach().cpu().numpy()
            # label_dtime, label_type = thin_predictions_or_labels(label_dtime, label_type)
            return (pred_dtime, pred_type), (label_dtime, label_type)

    def restore(self, ckpt_dir: str) -> None:
        """Load the checkpoint to restore the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """

        self.model.load_state_dict(torch.load(ckpt_dir), strict=False)
        if self.numeric_model:
            head, ext = os.path.splitext(ckpt_dir)
            numeric_ckpt_dir = f"{head}.numeric{ext}"
            self.numeric_model.load_state_dict(
                torch.load(numeric_ckpt_dir), strict=False
            )

    def save(self, ckpt_dir: str) -> None:
        """Save the checkpoint for the model.

        Args:
            ckpt_dir (str): path for the checkpoint.
        """
        torch.save(self.model.state_dict(), ckpt_dir)
        if self.numeric_model:
            head, ext = os.path.splitext(ckpt_dir)
            numeric_ckpt_dir = f"{head}.numeric{ext}"
            torch.save(self.numeric_model.state_dict(), numeric_ckpt_dir)
