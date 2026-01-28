import math
import pickle
from typing import Any, Optional

import torch
from easy_tpp.model.torch_model.torch_baselayer import (
    EncoderLayer,
    MultiHeadAttention,
    ScaledSoftplus,
)
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from torch import Tensor, nn

from patient_tpp.easy_tpp_zai.config_factory.model_config import ZaiModelConfig
from patient_tpp.easy_tpp_zai.model.torch_model.torch_thinning import ZaiEventSampler

MAX_LIFE_SPAN = 125


class ZaiAttNHP(TorchBaseModel):
    """Torch implementation of Attentive Neural Hawkes Process, ICLR 2022.
    https://arxiv.org/abs/2201.00044.
    Source code: https://github.com/yangalan123/anhp-andtt/blob/master/anhp/model/xfmr_nhp_fast.py

    Augmented with
        - invariant features (race, age, gender)
        - numerical features (BMI, HbA1C, ...)
        - clinical embeddings
    """

    def __init__(
        self,
        model_config: ZaiModelConfig,
        num_event_types_pad: Optional[int] = None,
        use_pad_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model

        Args:
            model_config (EasyTPP.ZaiModelConfig): config of model specs.
        """
        super(ZaiAttNHP, self).__init__(model_config)
        self.d_model = model_config.hidden_size
        self.use_norm = model_config.use_ln
        self.d_time = model_config.time_emb_size
        self.d_clinical = model_config.clinical_embedding_size
        self.kwargs = dict(kwargs)

        self.div_term = torch.exp(
            torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)
        ).reshape(1, 1, -1)

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        self.pseudoage_bins = model_config.pseudoage_bins
        self.pseudoage_embedding_size = model_config.pseudoage_embedding_size
        self.gender_cardinality = model_config.gender_cardinality
        self.gender_embedding_size = model_config.gender_embedding_size
        self.race_cardinality = model_config.race_cardinality
        self.race_embedding_size = model_config.race_embedding_size

        self.age_embedding = nn.Embedding(
            self.pseudoage_bins, self.pseudoage_embedding_size
        )
        self.gender_embedding = nn.Embedding(
            self.gender_cardinality, self.gender_embedding_size
        )
        self.race_embedding = nn.Embedding(
            self.race_cardinality, self.race_embedding_size
        )
        self.invar_affine = nn.Linear(
            self.gender_embedding_size
            + self.race_embedding_size
            + self.pseudoage_embedding_size,
            self.d_model,
        )

        self.pad_token_id = (
            model_config.pad_token_id if use_pad_token_id is None else use_pad_token_id
        )
        self.eps = torch.finfo(torch.float32).eps

        # TODO figure out how to get data spec config in here
        self.num_event_types_pad = (
            68 if num_event_types_pad is None else num_event_types_pad
        )
        self.layer_type_emb = nn.Embedding(
            self.num_event_types_pad, self.d_model, padding_idx=self.pad_token_id
        )
        if model_config.clinical_embeddings_path is None:
            clinical_embeddings = torch.zeros(self.num_event_types_pad, self.d_clinical)
        else:
            with open(model_config.clinical_embeddings_path, "rb") as fobj:
                clinical_embeddings = pickle.load(fobj)
        assert (
            clinical_embeddings.shape[-1] == self.d_clinical
        ), f"Size spec for clinical embeddings doesn't agree with asset \
(config {self.d_clinical}, asset {clinical_embeddings.shape[-1]}) import."
        self.clinical_embedding_layer = nn.Embedding.from_pretrained(
            clinical_embeddings,
            freeze=True,
            padding_idx=self.pad_token_id,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
        )
        self.clinical_affine = nn.Linear(128, self.d_clinical)

        self.headlist = []
        for _ in range(self.n_head):
            self.headlist.append(
                nn.ModuleList(
                    [
                        EncoderLayer(
                            self.d_model + self.d_time,
                            MultiHeadAttention(
                                1,
                                (1 * self.d_model) + (1 * self.d_time),
                                self.d_model,
                                self.dropout,
                                output_linear=False,
                            ),
                            use_residual=False,
                            dropout=self.dropout,
                        )
                        for _ in range(self.n_layers)
                    ]
                )
            )

        self.heads = nn.ModuleList(self.headlist)

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)
        self.inten_linear_1 = nn.Linear(
            # For main MHA...
            self.d_model * self.n_head  # For numeric model enc out...
            + self.d_model * self.n_head  # For clinical embeddings...
            + self.d_clinical,
            self.d_model * self.n_head,
        )
        self.inten_linear_2 = nn.Linear(
            self.d_model * self.n_head, self.num_event_types
        )
        self.softplus = ScaledSoftplus(
            self.num_event_types
        )  # learnable mark-specific beta
        self.layer_event_emb = nn.Linear(self.d_model + self.d_time, self.d_model)
        self.layer_intensity = nn.Sequential(
            self.inten_linear_1, self.inten_linear_2, self.softplus
        )

        if self.gen_config:
            self.event_sampler = ZaiEventSampler(
                num_sample=self.gen_config.num_sample,
                num_exp=self.gen_config.num_exp,
                over_sample_rate=self.gen_config.over_sample_rate,
                patience_counter=self.gen_config.patience_counter,
                num_samples_boundary=self.gen_config.num_samples_boundary,
                dtime_max=self.gen_config.dtime_max,
                device=self.device,
            )

    def compute_temporal_embedding(self, time: Tensor) -> Tensor:
        """Compute the temporal embedding.

        Args:
            time (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, emb_size].
        """
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)

        float_pe = pe.to(torch.float32)
        return float_pe

    def compute_invar_embedding(self, invars: Tensor, seq_len: int) -> Optional[Tensor]:
        """Build embedding for invariant features.

        Args:
            invars (tensor): [batch_size, ~3]
            seq_len (int): full sequence length

        Returns:
            tensor: [batch_size, seq_len, hidden_size]
        """
        invar_emb = None
        # if torch.Tensor.dim(invars) > 1:
        if invars is not None and (torch.Tensor.dim(invars) > 1):
            pseudoages = invars[:, 0] * (invars[:, 0] < MAX_LIFE_SPAN)
            discretized_ages = torch.floor(
                (pseudoages + 1) / self.pseudoage_bins
            ).long()
            age_emb = self.age_embedding(discretized_ages)
            gender_emb = self.gender_embedding(invars[:, 1].long() + 1)
            race_emb = self.race_embedding(invars[:, 2].long() + 1)
            invar_tr = self.invar_affine(
                torch.cat([age_emb, gender_emb, race_emb], dim=-1)
            )
            invar_emb = torch.zeros(invar_tr.shape[0], seq_len, invar_tr.shape[-1]).to(
                age_emb.device
            )
            # Set invariant features only for the first step in the sequence:
            invar_emb[:, 0, :] = invar_tr
            # invar_features = invar_features.unsqueeze(1).repeat(1, aug_features.shape[1], 1)
        return invar_emb

    def compute_clinical_embedding(
        self, event_seqs: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        all_clinical_embeds = self.clinical_embedding_layer(event_seqs)
        # clinical_emb = self.clinical_affine(all_clinical_embeds.sum(dim=-1))
        if seq_lens is None:
            # clinical_emb = nn.functional.normalize(all_clinical_embeds.sum(dim=-1))
            clinical_emb = all_clinical_embeds
        else:
            # clinical_emb = torch.transpose(torch.transpose(all_clinical_embeds.sum(dim=-1), 0, 1) / seq_lens, 0, 1)
            clinical_emb = all_clinical_embeds
        # return clinical_emb.unsqueeze(-1)
        return clinical_emb

    def forward_pass(
        self,
        init_cur_layer: Tensor,
        time_emb: Tensor,
        sample_time_emb: Tensor,
        event_emb: Tensor,
        combined_mask: Tensor,
        clinical_emb: Tensor,
        invar_emb: Optional[Tensor],
        numeric_layer: Optional[Tensor],
    ) -> Tensor:
        """update the structure sequentially.

        Args:
            init_cur_layer (tensor): [batch_size, seq_len, hidden_size]
            time_emb (tensor): [batch_size, seq_len, hidden_size]
            sample_time_emb (tensor): [batch_size, seq_len, hidden_size]
            event_emb (tensor): [batch_size, seq_len, hidden_size]
            combined_mask (tensor): [batch_size, 4*seq_len, hidden_size]
            invar_emb (tensor): [batch_size, seq_len, hidden_size]
            numeric_layer (tensor): [batch_size, seq_len, hidden_size] Supplemental layer computed
            on its own forward pass
        Returns:
            tensor: [batch_size, seq_len, hidden_size*2]
        """
        cur_layers = []
        seq_len = event_emb.size(1)
        for head_i in range(self.n_head):
            # [batch_size, seq_len, hidden_size]
            cur_layer_ = init_cur_layer
            for layer_i in range(self.n_layers):
                # each layer concats the temporal emb
                # [batch_size, seq_len, hidden_size + d_time]
                layer_ = torch.cat(
                    [cur_layer_.to(sample_time_emb.device), sample_time_emb], dim=-1
                )

                # make combined input from event emb + layer emb
                # [batch_size, seq_len*2, hidden_size + d_time]
                _combined_input = torch.cat([event_emb, layer_], dim=1)
                enc_layer = self.headlist[head_i][layer_i]
                # compute the output ... this invokes the EncoderLayer's forward method,
                # which calls a MultiHeadAttention object with the combined_input as key,
                # value, and query.
                enc_output = enc_layer(_combined_input, combined_mask)

                # the layer output
                # [batch_size, seq_len, hidden_size]
                _cur_layer_ = enc_output[:, seq_len:, :]
                # add residual connection
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_

                if invar_emb is not None:
                    # Add invariant features to layer (only affects beginning of sequence
                    # by construction)
                    cur_layer_ = cur_layer_ + invar_emb

                # event emb
                event_emb = torch.cat([_cur_layer_, time_emb], dim=-1)

                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        if numeric_layer is not None:
            cur_layers.append(numeric_layer)
        cur_layers.append(clinical_emb)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    def seq_encoding(
        self,
        time_seqs: Tensor,
        event_seqs: Tensor,
        embedding_layer: Optional[nn.Embedding],
        numeric_layer: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Encode the sequence.

        Args:
            time_seqs (tensor): time seqs input, [batch_size, seq_len].
            event_seqs (_type_): event type seqs input, [batch_size, seq_len].

        Returns:
            tuple: event embedding, time embedding, type embedding, clinical embedding, and numeric embedding.
        """
        if embedding_layer is None:
            embedding_layer = self.layer_type_emb

        # [batch_size, seq_len, hidden_size]
        time_emb = self.compute_temporal_embedding(time_seqs)
        # [batch_size, seq_len, hidden_size]
        type_emb = torch.tanh(embedding_layer(event_seqs.long()))
        if numeric_layer is None:
            # NB This assumes numeric hidden layer size = indicative hidden layer size
            numeric_layer = torch.zeros(
                event_seqs.shape[0], event_seqs.shape[1], 2 * self.d_model
            ).to(type_emb.device)
        clinical_emb = self.compute_clinical_embedding(event_seqs.long())
        # [batch_size, seq_len, hidden_size*2 + pretrained_size]
        event_emb = torch.cat([type_emb, time_emb], dim=-1)

        return event_emb, time_emb, type_emb, clinical_emb, numeric_layer

    def make_layer_mask(self, attention_mask: Tensor) -> Tensor:
        """Create a tensor to do masking on layers.

        Args:
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]

        Returns:
            tensor: aim to keep the current layer, the same size of attention mask
            a diagonal matrix, [batch_size, seq_len, seq_len]
        """
        # [batch_size, seq_len, seq_len]
        layer_mask = (
            (torch.eye(attention_mask.size(1), device=self.device) < 1)
            .unsqueeze(0)
            .expand_as(attention_mask)
        )
        return layer_mask

    def make_combined_att_mask(
        self, attention_mask: Tensor, layer_mask: Tensor
    ) -> Tensor:
        """Combined attention mask and layer mask.

        Args:
            attention_mask (tensor): mask for attention operation, [batch_size, seq_len, seq_len]
            layer_mask (tensor): mask for other layers, [batch_size, seq_len, seq_len]

        Returns:
            tensor: [batch_size, seq_len * 2, seq_len * 2]
        """
        # [batch_size, seq_len, seq_len * 2]
        combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # [batch_size, seq_len, seq_len * 2]
        contextual_mask = torch.cat(
            [attention_mask, torch.ones_like(layer_mask)], dim=-1
        )
        # [batch_size, seq_len * 2, seq_len * 2]
        combined_mask = torch.cat([contextual_mask, combined_mask], dim=1)
        return combined_mask

    def forward(
        self,
        time_seqs: Tensor,
        event_seqs: Tensor,
        attention_mask: Tensor,
        invars: Tensor,
        numeric_layer: Optional[Tensor],
        sample_times: Optional[Tensor] = None,
    ) -> Tensor:
        """Call the model.

        Args:
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            event_seqs (tensor): [batch_size, seq_len], sequences of event types.
            attention_mask (tensor): [batch_size, seq_len, seq_len], masks for event sequences.
            invars (tensor): [batch_size, ~3], invariant features associated with sequences.
            numeric_layer (tensor): [batch_size, seq_len, hidden_size] Supplemental layer computed
                on its own forward pass
            sample_times (tensor, optional): [batch_size, seq_len, num_samples]. Defaults to None.

        Returns:
            tensor: states at sampling times, [batch_size, seq_len, num_samples].
        """
        event_emb, time_emb, type_emb, clinical_emb, numeric_layer = self.seq_encoding(
            time_seqs, event_seqs, None, numeric_layer=numeric_layer
        )
        init_cur_layer = torch.zeros_like(type_emb).to(event_emb.device)
        if sample_times is None:
            sample_time_emb = time_emb
        else:
            sample_time_emb = self.compute_temporal_embedding(sample_times)
        invar_emb = self.compute_invar_embedding(invars, time_seqs.shape[1])

        layer_mask = self.make_layer_mask(attention_mask)
        combined_mask = self.make_combined_att_mask(attention_mask, layer_mask)
        cur_layer_ = self.forward_pass(
            init_cur_layer,
            time_emb,
            sample_time_emb,
            event_emb,
            combined_mask,
            clinical_emb,
            invar_emb,
            numeric_layer=numeric_layer,
        )

        return cur_layer_

    def loglike_loss(
        self, batch: list[Tensor], numeric_layer: Optional[Tensor] = None
    ) -> tuple[float, int, Tensor]:
        """Compute the loglike loss.

        Args:
            batch (list): batch input.
            numeric_layer (tensor): [batch_size, seq_len, hidden_size] Supplemental layer computed
                on its own forward pass

        Returns:
            list: loglike loss, num events.
        """
        (
            time_seqs,
            time_delta_seqs,
            type_seqs,
            batch_non_pad_mask,
            attention_mask,
            _,
            invars,
            _,
        ) = batch

        # seq_lens = torch.sum(batch_non_pad_mask, axis=1)  # (don't know how to vectorize these yet)

        # 1. compute event-loglike
        # the prediction of last event has no label, so we proceed to the last but one
        # att mask => diag is False, not mask.
        enc_out = self.forward(
            time_seqs[:, :-1],
            type_seqs[:, :-1],
            attention_mask[:, :-1, :-1],
            invars,
            numeric_layer=numeric_layer,
            sample_times=time_seqs[:, 1:],
        ).to(time_seqs.device)
        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.layer_intensity(enc_out)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        temp_time = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # [batch_size, seq_len, num_sample]
        sample_times = temp_time + time_seqs[:, :-1].unsqueeze(-1)

        # 2.2 compute intensities at sampled times
        # [batch_size, seq_len = max_len - 1, num_sample, event_num]
        lambda_t_sample = self.compute_intensities_at_sample_times(
            time_seqs[:, :-1],
            time_delta_seqs[:, :-1],  # not used
            type_seqs[:, :-1],
            invars,
            sample_times,
            attention_mask=attention_mask[:, :-1, :-1],
            numeric_layer=numeric_layer,
        )

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=type_seqs[:, 1:],
        )

        # compute loss to minimize
        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events, enc_out

    def compute_states_at_sample_times(
        self,
        time_seqs: Tensor,
        type_seqs: Tensor,
        attention_mask: Tensor,
        invars: Tensor,
        sample_times: Tensor,
        numeric_layer: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the states at sampling tim
        Args:
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            time_delta_seqs (tensor): [batch_size, seq_len], sequences of delta times.
            type_seqs (tensor): [batch_size, seq_len], sequences of event types.
            attention_mask (tensor): [batch_size, seq_len, seq_len], masks for event sequences.
            invars (tensor): [batch_size, ~3], invariant features associated with sequences.
            sample_times (tensor): delta times in sampling.
            numeric_layer (tensor): [batch_size, seq_len, hidden_size] Supplemental layer computed
                on its own forward pass

        Returns:
            tensor: hiddens states at sampling times.
        """
        batch_size = type_seqs.size(0)
        seq_len = type_seqs.size(1)
        num_samples = sample_times.size(-1)

        # [num_samples, batch_size, seq_len]
        sample_times = sample_times.permute((2, 0, 1))
        # [num_samples * batch_size, seq_len]
        _sample_time = sample_times.reshape(num_samples * batch_size, -1)
        # [num_samples * batch_size, seq_len]
        _types = type_seqs.expand(num_samples, -1, -1).reshape(
            num_samples * batch_size, -1
        )
        # [num_samples * batch_size, seq_len]
        _times = time_seqs.expand(num_samples, -1, -1).reshape(
            num_samples * batch_size, -1
        )
        # [num_samples * batch_size, ~3]
        _invars = torch.empty(invars.size())
        if torch.Tensor.dim(invars) > 1:
            _invars = invars.expand(num_samples, -1, -1).reshape(
                num_samples * batch_size, -1
            )
        if numeric_layer is not None:
            numeric_layer = (
                numeric_layer.unsqueeze(0)
                .expand(num_samples, -1, -1, -1)
                .reshape(num_samples * batch_size, -1, 2 * self.d_model)
            )
        # [num_samples * batch_size, seq_len]
        _attn_mask = (
            attention_mask.unsqueeze(0)
            .expand(num_samples, -1, -1, -1)
            .reshape(num_samples * batch_size, seq_len, seq_len)
        )
        # time_seqs, event_seqs, attention_mask, and sample_times
        # [num_samples * batch_size, seq_len, hidden_size]
        encoder_output = self.forward(
            _times, _types, _attn_mask, _invars, numeric_layer, _sample_time
        )

        # [num_samples, batch_size, seq_len, hidden_size]
        encoder_output = encoder_output.reshape(num_samples, batch_size, seq_len, -1)
        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = encoder_output.permute((1, 2, 0, 3))
        return encoder_output

    def compute_intensities_at_sample_times(
        self,
        time_seqs: Tensor,
        _time_delta_seqs: Tensor,
        type_seqs: Tensor,
        invars: Tensor,
        sample_dtimes: Tensor,
        numeric_layer: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the intensity at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], sequences of timestamps.
            time_delta_seqs (tensor): [batch_size, seq_len], sequences of delta times.
            type_seqs (tensor): [batch_size, seq_len], sequences of event types.
            invars (tensor): [batch_size, ~3], invariant features associated with sequences.
            sampled_dtimes (tensor): [batch_size, seq_len, num_sample], sampled time delta sequence.
            numeric_layer (tensor): [batch_size, seq_len, hidden_size] Supplemental layer computed
                on its own forward pass

        Returns:
            tensor: intensities as sampled_dtimes, [batch_size, seq_len, num_samples, event_num].
        """
        attention_mask = kwargs.get("attention_mask")
        compute_last_step_only = kwargs.get("compute_last_step_only", False)
        # seq_lens = kwargs.get('seq_lens', None)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = (
                torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
                .unsqueeze(0)
                .to(type_seqs.device)
            )
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        if sample_dtimes.size()[1] < time_seqs.size()[1]:
            # we pass sample_dtimes for last time step here
            # we do a temp solution
            # [batch_size, seq_len, num_samples]
            sample_dtimes = time_seqs[:, :, None] + torch.tile(
                sample_dtimes, [1, time_seqs.size()[1], 1]
            )

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(
            time_seqs,
            type_seqs,
            attention_mask,
            invars,
            sample_dtimes,
            numeric_layer=numeric_layer,
        ).to(time_seqs.device)

        if compute_last_step_only:
            lambdas = self.layer_intensity(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.layer_intensity(encoder_output)
        return lambdas

    def predict_one_step_at_every_event(
        self, batch: list[Tensor], numeric_layer: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One-step prediction for every event in the sequence.

        Args:
            batch, including:
                time_seqs (tensor): [batch_size, seq_len].
                time_delta_seqs (tensor): [batch_size, seq_len].
                event_seq (tensor): [batch_size, seq_len].
                batch_non_pad_mask (tensor): [batch_size, seq_len].
                seq_len
            numeric_layer: Already-computed encoding of numeric features

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _, _, invars, _ = batch

        # Get actual sequence lengths from mask
        # seq_lens = torch.sum(batch_non_pad_mask, axis=1)  # (don't know how to vectorize these yet)

        # Grab the last n items in each sequence (so we're not making predictions on padding)
        # Assumes right padding.
        # indexed_mask = batch_non_pad_mask * torch.arange(start=1, end=batch_non_pad_mask.shape[1] + 1,\
        #     step=1).to(device=batch_non_pad_mask.device)
        # _, topk_indices = torch.topk(input=indexed_mask, k=indexed_mask.shape[1], dim=-1, largest=False, sorted=True)
        # topk_indices = torch.flip(topk_indices, dims=[1])

        # remove the last event, as the prediction based on the last event has no label
        # note: the first dts is 0
        # [batch_size, seq_len]
        time_seq, time_delta_seq, event_seq = (
            time_seq[:, :-1],
            time_delta_seq[:, :-1],
            event_seq[:, :-1],
        )
        # time_seq = torch.gather(time_seq, 1, topk_indices)[:, :-1]
        # time_delta_seq = torch.gather(time_seq, 1, topk_indices)[:, :-1]
        # event_seq = torch.gather(event_seq, 1, topk_indices)[:, :-1]

        # [batch_size, seq_len]

        dtime_boundary = torch.max(
            time_delta_seq * self.event_sampler.dtime_max,
            time_delta_seq + self.event_sampler.dtime_max,
        )
        # dtime_boundary = torch.max(time_delta_seq + self.event_sampler.dtime_max)

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
            time_seq,
            time_delta_seq,
            event_seq,
            invars,
            dtime_boundary,
            self.compute_intensities_at_sample_times,
            compute_last_step_only=False,
            numeric_layer=numeric_layer,
        )  # make it explicit

        # We should condition on each accepted time to sample event mark,
        # but not condition on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(
            time_seq,
            time_delta_seq,
            event_seq,
            invars,
            accepted_dtimes,
            numeric_layer=numeric_layer,
        )

        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(
            dim=-1, keepdim=True
        )

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum(
            "...s,...sm->...m", weights, intensities_normalized
        )

        # [batch_size, seq_len]
        types_pred = torch.argmax(intensities_weighted, dim=-1)
        # [batch_size, seq_len]
        dtimes_pred = torch.sum(
            accepted_dtimes * weights, dim=-1
        )  # compute the expected next event time
        return dtimes_pred, types_pred

    def predict_multi_step_since_last_event(
        self,
        batch: list[Tensor],
        numeric_layer: Tensor,
        forward: bool = False,
        return_intensities: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Multi-step prediction since last event in the sequence.

        Args:
            batch (tensor): [batch_size, seq_len].
            numeric_layer (tensor): [batch_size, numeric_seq_len, num_numeric_events]
            time_delta_seqs (tensor): [batch_size, seq_len]
            type_seqs (tensor): [batch_size, seq_len]

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
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
        # Get sequence lengths from mask (these are all nominal when using the label mask)
        # seq_lens = torch.sum(batch_non_pad_mask_label, axis=1)

        num_step = min(self.gen_config.num_step_gen, event_seq_label.shape[-1] - 1)

        # Grab the last n items in each sequence (so we're not making predictions on padding)
        # Assumes right padding.
        # indexed_mask = batch_non_pad_mask_label * torch.arange(start=1, end=batch_non_pad_mask_label.shape[1] + 1,\
        #     step=1).to(device=batch_non_pad_mask_label.device)
        # _, topk_indices = torch.topk(input=indexed_mask, k=max(obj_lens), dim=-1, largest=False, sorted=True)
        # topk_indices = torch.flip(topk_indices, dims=[1]) flip for largest=True

        if not forward:
            time_seq = time_seq_label[:, :-num_step]
            time_delta_seq = time_delta_seq_label[:, :-num_step]
            event_seq = event_seq_label[:, :-num_step]
        else:
            time_seq, time_delta_seq, event_seq = (
                time_seq_label,
                time_delta_seq_label,
                event_seq_label,
            )

        for i in range(num_step):
            # [batch_size, seq_len]
            dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

            # [batch_size, 1, num_sample]
            numeric_layer_arg = (
                None if numeric_layer is None else numeric_layer[:, : -num_step + i, :]
            )
            accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
                time_seq,
                time_delta_seq,
                event_seq,
                invars,
                dtime_boundary,
                self.compute_intensities_at_sample_times,
                compute_last_step_only=True,
                numeric_layer=numeric_layer_arg,
            )

            # [batch_size, 1]
            dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

            # [batch_size, seq_len, 1, event_num]
            intensities_at_times = self.compute_intensities_at_sample_times(
                time_seq,
                time_delta_seq,
                event_seq,
                invars,
                dtimes_pred[:, :, None],
                max_steps=event_seq.size()[1],
                numeric_layer=numeric_layer_arg,
            )

            # [batch_size, seq_len, event_num]
            intensities_at_times = intensities_at_times.squeeze(dim=-2)
            # [batch_size, seq_len]
            types_pred = torch.argmax(intensities_at_times, dim=-1)

            types_pred_ = types_pred[:, -1:]
            # [batch_size, 1]
            dtimes_pred_ = dtimes_pred[:, -1:]
            time_pred_ = time_seq[:, -1:] + dtimes_pred_

            # concat to the prefix sequence
            time_seq = torch.cat([time_seq, time_pred_], dim=-1)
            time_delta_seq = torch.cat([time_delta_seq, dtimes_pred_], dim=-1)
            event_seq = torch.cat([event_seq, types_pred_], dim=-1)

        event_seq_rval = (
            intensities_at_times[:, -num_step:]
            if return_intensities
            else event_seq[:, -num_step:]
        )

        return (
            time_delta_seq[:, -num_step:],
            event_seq_rval,
            time_delta_seq_label[:, -num_step:],
            event_seq_label[:, -num_step:],
        )


class ZaiNumericAttNHP(ZaiAttNHP):
    """
    Numeric feature variant; numeric sequences are packed with several (~8) features
    in parallel (i.e. values sharing a timestamp).
    """

    def __init__(
        self,
        model_config: ZaiModelConfig,
        num_event_types_pad: Optional[int] = None,
        use_pad_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model

        Args:
            model_config (EasyTPP.ZaiModelConfig): config of model specs.
        """
        super(ZaiNumericAttNHP, self).__init__(
            model_config,
            num_event_types_pad=num_event_types_pad,
            use_pad_token_id=use_pad_token_id,
            **kwargs,
        )
        # TODO figure out how to get data spec config in here
        self.pad_token_id = 41 if use_pad_token_id is None else use_pad_token_id
        self.num_event_types_pad = (
            44 if num_event_types_pad is None else num_event_types_pad
        )
        self.num_tracks = kwargs.get("num_tracks", 8)
        self.layer_type_emb = nn.Embedding(
            self.num_event_types_pad, self.d_model, padding_idx=self.pad_token_id
        )

        del self.clinical_embedding_layer
        del self.heads
        self.headlist = []
        for _ in range(self.n_head):
            self.headlist.append(
                nn.ModuleList(
                    [
                        EncoderLayer(
                            1 * self.d_model + self.d_time,
                            MultiHeadAttention(
                                1,
                                (1 * self.d_model) + (1 * self.d_time),
                                self.d_model,
                                self.dropout,
                                output_linear=False,
                            ),
                            use_residual=False,
                            dropout=self.dropout,
                        )
                        for _ in range(self.n_layers)
                    ]
                )
            )

        self.heads = nn.ModuleList(self.headlist)
        self.inten_linear = nn.Linear(self.d_model * self.n_head, self.num_event_types)
        self.layer_event_emb = nn.Linear(self.d_model + self.d_time, self.d_model)
        self.layer_intensity = nn.Sequential(self.inten_linear, self.softplus)

    def forward_pass(
        self,
        init_cur_layer: Tensor,
        time_emb: Tensor,
        sample_time_emb: Tensor,
        event_emb: Tensor,
        combined_mask: Tensor,
        _clinical_emb: Tensor,
        _invar_emb: Optional[Tensor],
        numeric_layer: Optional[Tensor],  # noqa: ARG002
    ) -> Tensor:
        """update the structure sequentially.

        Args:
            init_cur_layer (tensor): [batch_size, seq_len, hidden_size]
            time_emb (tensor): [batch_size, seq_len, hidden_size]
            sample_time_emb (tensor): [batch_size, seq_len, hidden_size]
            event_emb (tensor): [batch_size, seq_len, hidden_size]
            combined_mask (tensor): [batch_size, 4*seq_len, hidden_size]
            _clinical_emb (tensor): [batch_size, seq_len]
            _invar_emb (tensor): [batch_size, seq_len, 3]
        Returns:
            tensor: [batch_size, seq_len, hidden_size*2]
        """
        cur_layers = []
        seq_len = event_emb.size(1)
        for head_i in range(self.n_head):
            # [batch_size, seq_len, hidden_size]
            cur_layer_ = init_cur_layer
            for layer_i in range(self.n_layers):
                # each layer concats the temporal emb
                # [batch_size, seq_len, hidden_size + d_time]
                layer_ = torch.cat(
                    [cur_layer_.to(sample_time_emb.device), sample_time_emb], dim=-1
                )

                # make combined input from event emb + layer emb
                # [batch_size, seq_len*2, hidden_size + d_time]
                _combined_input = torch.cat([event_emb, layer_], dim=1)
                enc_layer = self.headlist[head_i][layer_i]
                # compute the output
                enc_output = enc_layer(_combined_input, combined_mask)

                # the layer output
                # [batch_size, seq_len, hidden_size]
                _cur_layer_ = enc_output[:, seq_len:, :]
                # add residual connection
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_

                # event emb
                event_emb = torch.cat([_cur_layer_, time_emb], dim=-1)

                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    def seq_encoding(
        self,
        time_seqs: Tensor,
        event_seqs: Tensor,
        embedding_layer: Optional[nn.Embedding],
        numeric_layer: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:  # noqa: ARG002
        """Encode the sequence.
        Args:
            time_seqs (tensor): time seqs input, [batch_size, seq_len].
            event_seqs (tensor): event type seqs input, [batch_size, seq_len].
            embedding_layer (nn.Embedding): encoder for event types

        Returns:
            tuple: event embedding, time embedding and type embedding.
        """
        if embedding_layer is None:
            embedding_layer = self.layer_type_emb
        # [batch_size, seq_len, hidden_size]
        time_emb = self.compute_temporal_embedding(time_seqs)
        # [batch_size, seq_len, hidden_size]
        type_emb = torch.tanh(embedding_layer(event_seqs.long()))
        # [batch_size, seq_len, hidden_size*2 + pretrained_size]
        # time_emb = time_emb.unsqueeze(-2).repeat(1, 1, type_emb.shape[2], 1)
        event_emb = torch.cat([type_emb, time_emb], dim=-1)

        return event_emb, time_emb, type_emb, torch.empty(event_emb.size()), None

    def loglike_loss(
        self, batch: list[Tensor], numeric_layer: Optional[Tensor] = None
    ) -> tuple[float, int, Tensor]:  # noqa: ARG002
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        if batch is None:
            return 0, 0, None

        (
            time_seqs,
            time_delta_seqs,
            type_seqs,
            batch_non_pad_mask,
            attention_mask,
            _,
            invars,
            _,
        ) = batch

        loss = 0
        num_events = 0
        # seq_lens = torch.sum(batch_non_pad_mask, axis=1)  # (don't know how to vectorize these yet)

        # 1. compute event-loglike
        # the prediction of last event has no label, so we proceed to the last but one
        # att mask => diag is False, not mask.
        enc_out = self.forward(
            time_seqs[:, :-1],
            type_seqs[:, :-1],
            attention_mask[:, :-1, :-1],
            invars,
            None,
            sample_times=time_seqs[:, 1:],
        ).to(time_seqs.device)
        # [batch_size, seq_len, num_event_types]
        lambda_at_event = self.layer_intensity(enc_out)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # [batch_size, seq_len, num_sample]
        temp_time = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # [batch_size, seq_len, num_sample]
        sample_times = temp_time + time_seqs[:, :-1].unsqueeze(-1)

        # 2.2 compute intensities at sampled times
        # [batch_size, seq_len = max_len - 1, num_sample, event_num]
        lambda_t_sample = self.compute_intensities_at_sample_times(
            time_seqs[:, :-1],
            time_delta_seqs[:, :-1],  # not used
            type_seqs[:, :-1],
            invars,
            sample_times,
            attention_mask=attention_mask[:, :-1, :-1],
            numeric_layer=None,
        )

        event_ll, non_event_ll, an_event_count = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=type_seqs[:, 1:].long(),
        )
        num_events += an_event_count

        # compute loss to minimize
        loss += -((event_ll - non_event_ll).sum())
        return loss, num_events, enc_out

    def predict_multi_step_since_last_event(
        self,
        batch: list[Tensor],
        _numeric_layer: Tensor,
        forward: bool = False,
        return_intensities: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Multi-step prediction since last event in the sequence.

        Args:
            batch (tensor): [batch_size, seq_len].
            numeric_batch (tensor): [batch_size, numeric_seq_len, num_numeric_events]
            forward (bool): If False, run in backtesting mode (working forward from num_step steps from
                the end)
            return_intensities: If True, return raw score as opposed to the argmaxes

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
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
        # Get sequence lengths from mask (these are all nominal when using the label mask)
        # seq_lens = torch.sum(batch_non_pad_mask_label, axis=1)

        num_step = min(self.gen_config.num_step_gen, event_seq_label.shape[-1] - 1)

        if not forward:
            time_seq = time_seq_label[:, :-num_step]
            time_delta_seq = time_delta_seq_label[:, :-num_step]
            event_seq = event_seq_label[:, :-num_step]
        else:
            time_seq, time_delta_seq, event_seq = (
                time_seq_label,
                time_delta_seq_label,
                event_seq_label,
            )

        for _ in range(num_step):
            # [batch_size, seq_len]
            dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

            # [batch_size, 1, num_sample]
            accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
                time_seq,
                time_delta_seq,
                event_seq,
                invars,
                dtime_boundary,
                self.compute_intensities_at_sample_times,
                compute_last_step_only=True,
                numeric_layer=None,
            )

            # [batch_size, 1]
            dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

            # [batch_size, seq_len, 1, event_num]
            intensities_at_times = self.compute_intensities_at_sample_times(
                time_seq,
                time_delta_seq,
                event_seq,
                invars,
                dtimes_pred[:, :, None],
                max_steps=event_seq.size()[1],
                numeric_layer=None,
            )

            # [batch_size, seq_len, event_num]
            intensities_at_times = intensities_at_times.squeeze(dim=-2)
            # [batch_size, seq_len]
            types_pred = torch.argmax(intensities_at_times, dim=-1)

            # intensities_pred = intensities_at_times[:, [-(num_step - i)], :]
            types_pred_ = types_pred[:, -1:]
            # [batch_size, 1]
            dtimes_pred_ = dtimes_pred[:, -1:]
            time_pred_ = time_seq[:, -1:] + dtimes_pred_

            # concat to the prefix sequence
            time_seq = torch.cat([time_seq, time_pred_], dim=-1)
            time_delta_seq = torch.cat([time_delta_seq, dtimes_pred_], dim=-1)
            event_seq = torch.cat([event_seq, types_pred_], dim=-1)

        event_seq_rval = (
            intensities_at_times[:, -num_step:]
            if return_intensities
            else event_seq[:, -num_step:]
        )

        return (
            time_delta_seq[:, -num_step:],
            event_seq_rval,
            time_delta_seq_label[:, -num_step:],
            event_seq_label[:, -num_step:],
        )
