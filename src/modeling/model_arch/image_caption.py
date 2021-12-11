import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.modeling.model_arch.conv_models import LuxResNet


class SimpleLSTMModel(nn.Module):
    def __init__(self, **kwargs):
        """
        from
        https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html
        """
        super().__init__()
        if kwargs["encoder"] == "imagenet":
            raise NotImplementedError
            # self.encoder = LuxEncoder(
            #     timm_params=kwargs["timm_params"],
            #     gem_power=kwargs["gem_power"],
            #     gem_requires_grad=kwargs["gem_requires_grad"],
            #     in_channels=kwargs["in_channels"],
            #     num_classes=kwargs["num_classes"],
            #     out_indices=kwargs["encoder_out_indices"],
            # )
            encoder_hidden_dim = int(2 ** (5 + max(kwargs["encoder_out_indices"])))
        else:
            encoder_hidden_dim = kwargs["encoder_hidden_dim"]
            self.encoder = LuxResNet(
                in_channels=kwargs["in_channels"],
                filters=encoder_hidden_dim,
                remove_head=True,
                use_point_conv=kwargs["use_point_conv"],
            )

        self.decoder = SimpleLSTMDecoder(
            encoder_hidden_dim=encoder_hidden_dim,
            embed_dim=kwargs["decoder"]["embed_dim"],
            in_features=kwargs["decoder"]["in_features"],
            out_features=kwargs["decoder"]["out_features"],
            hidden_dim=kwargs["decoder"]["hidden_dim"],
            encoder_downsampled_dim=encoder_hidden_dim * 2
            if kwargs["use_point_conv"]
            else encoder_hidden_dim * 4,
            input_feed_size=kwargs["decoder"]["input_feed_size"],
            att_hid_size=kwargs["decoder"]["att_hid_size"],
            use_pooled_feat=kwargs["decoder"]["use_pooled_feat"],
        )

    # def forward(self, src_tokens, src_lengths, prev_output_tokens):
    def forward(
        self,
        inputs: torch.Tensor,
        aux_inputs: Optional[torch.Tensor] = None,
    ):
        # We could override the ``forward()`` if we wanted more control over how
        # the encoder and decoder interact, but it's not necessary for this
        # tutorial since we can inherit the default implementation provided by
        # the FairseqEncoderDecoderModel base class, which looks like:
        #
        # encoder_out = self.encoder(src_tokens, src_lengths)
        encoder_out = self.encoder(inputs)
        decoder_out = self.decoder(aux_inputs, encoder_out)
        return decoder_out


class SimpleLSTMDecoder(nn.Module):
    def __init__(
        self,
        in_features: int = 4,
        encoder_hidden_dim=128,
        embed_dim=32,
        hidden_dim=128,
        dropout=0.1,
        lstm_dropout=0.1,
        out_features: int = 5,
        encoder_downsampled_dim: int = 128,
        input_feed_size: int = 128,
        att_hid_size: int = 64,
        use_pooled_feat: bool = False,
    ):
        """
        from
        https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        # input_feed_size = 0 if encoder_output_units == 0 else hidden_size

        self.hidden_size = hidden_dim
        self.input_feed_size = input_feed_size

        self.use_pooled_feat = use_pooled_feat
        num_layers = 1
        input_size = self.input_feed_size + encoder_hidden_dim + embed_dim

        self.embed_tokens = nn.Linear(in_features, embed_dim)
        if self.use_pooled_feat:
            self.pool = nn.AdaptiveAvgPool2d(1)
            input_size += encoder_hidden_dim

        self.layers = nn.ModuleList(
            [
                nn.LSTMCell(
                    input_size=input_size,
                    hidden_size=hidden_dim,
                )
                for layer in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.dropout_out_module = nn.Dropout(p=lstm_dropout)

        self.encoder_downsampled_dim = encoder_downsampled_dim

        if att_hid_size == 0:
            self.attention = None
        else:
            self.attention = Attention(rnn_size=hidden_dim, att_hid_size=att_hid_size)
            self.encoder_feat_projection = nn.Linear(
                self.encoder_downsampled_dim, att_hid_size
            )
            hidden_dim = hidden_dim + encoder_downsampled_dim

        if input_feed_size == 0:
            self.output_projection = nn.Linear(hidden_dim, out_features)
        else:
            assert self.attention is not None
            self.after_attention_projection = nn.Linear(hidden_dim, input_feed_size)
            self.output_projection = nn.Linear(input_feed_size, out_features)

    # We now take an additional kwarg (*incremental_state*) for caching the
    # previous hidden and cell states.
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            # If the *incremental_state* argument is not ``None`` then we are
            # in incremental inference mode. While *prev_output_tokens* will
            # still contain the entire decoded prefix, we will only use the
            # last step and assume that the rest of the state is cached.
            prev_output_tokens = prev_output_tokens[:, -1:]

        # This remains the same as before.
        bsz, tgt_len, _ = prev_output_tokens.size()
        final_encoder_hidden = encoder_out["feature"]

        # input_sequence = self.get_input_sequence(
        #     bsz, tgt_len, xy_posi.clone(), final_encoder_hidden, encoder_hidden_dim
        # )

        # self._random_feat_check(
        #     bsz, xy_posi, final_encoder_hidden, input_sequence, num_checks=1000
        # )
        x = prev_output_tokens
        x = self.embed_tokens(x)
        if self.use_pooled_feat:
            pooled_feat = self.pool(encoder_out["feature"]).squeeze()
            x = torch.cat(
                [
                    x,
                    pooled_feat.unsqueeze(1).expand(bsz, tgt_len, -1),
                ],
                dim=2,
            )
        # x = self.dropout(x)

        # # We will now check the cache and load the cached previous hidden and
        # # cell states, if they exist, otherwise we will initialize them to
        # # zeros (as before). We will use the ``utils.get_incremental_state()``
        # # and ``utils.set_incremental_state()`` helpers.
        # initial_state = utils.get_incremental_state(
        #     self,
        #     incremental_state,
        #     "prev_state",
        # )
        initial_state = None
        # if initial_state is None:
        #     # first time initialization, same as the original version
        #     initial_state = (
        #         final_encoder_hidden.unsqueeze(0),  # hidden
        #         torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
        #     )

        # bsz, hiddim_dim, positional_sequence -> bsz,positional_sequencem, hiddim_dim,
        encoder_feats = (
            encoder_out["final_hidden"]
            .view(bsz, self.encoder_downsampled_dim, -1)
            .transpose(2, 1)
        )
        projected_encoder_feats = None
        if self.attention is not None:
            projected_encoder_feats = self.encoder_feat_projection(encoder_feats)

        # Run one step of our LSTM.
        # output, latest_state = self.lstm(x.transpose(0, 1), initial_state)
        x, _ = self.extract_features(
            x=x,
            prev_output_tokens=prev_output_tokens,
            final_encoder_hidden=final_encoder_hidden,
            encoder_out=None,
            encoder_feats=encoder_feats,
            projected_encoder_feats=projected_encoder_feats,
        )

        # # Update the cache with the latest hidden and cell states.
        # utils.set_incremental_state(
        #     self,
        #     incremental_state,
        #     "prev_state",
        #     latest_state,
        # )

        # This remains the same as before
        # x = output.transpose(0, 1)
        # x = self.lstm_dropout(x)
        x = self.output_projection(x)
        # return x, None
        return {"outputs": x}

    def get_input_sequence(
        self,
        bsz: int,
        tgt_len: int,
        xy_posi: torch.Tensor,
        final_encoder_hidden: torch.Tensor,
        encoder_hidden_dim: int,
    ):
        if (bsz == 1) & (tgt_len == 1):
            return final_encoder_hidden[:, :, xy_posi[..., 1], xy_posi[..., 0]].squeeze(
                -1
            )

        # xy_posi (bsz, tgt_len, 2)
        # final_encoder_hidden (bsz, encoder_hidden_dim, 32, 32)
        xy_posi = xy_posi[..., 1] * final_encoder_hidden.size(-2) + xy_posi[..., 0]
        # (bsz, encoder_hidden_dim, 32x32)
        xy_feats = final_encoder_hidden.view(bsz, final_encoder_hidden.size(1), -1)

        if tgt_len == 1:
            xy_ind = xy_posi.unsqueeze(-1).expand(bsz, encoder_hidden_dim).unsqueeze(-1)
        else:
            xy_ind = (
                xy_posi.unsqueeze(-1)
                .expand(bsz, tgt_len, encoder_hidden_dim)
                .unsqueeze(-1)
            )
            xy_feats = xy_feats.unsqueeze(1).expand(
                bsz, tgt_len, encoder_hidden_dim, -1
            )

        # (bsz, tag_len, encoder_hidden_dim)
        # input_sequence = (xy_feats * xy_onehot).sum(-1)
        input_sequence = xy_feats.gather(-1, xy_ind).squeeze(-1)
        # assert ((lazy == input_sequence).sum() == 0).item()
        return input_sequence

    def _random_feat_check(
        self, bsz, xy_posi, final_encoder_hidden, input_sequence, num_checks: int = 1000
    ):
        for _ in range(num_checks):
            max_len_for_each_batch = (xy_posi != -1)[..., 0].sum(dim=-1)
            ba_ind = random.choice(range(bsz))
            seq_ind = random.choice(range(max_len_for_each_batch[ba_ind].item()))
            xy = xy_posi[ba_ind, seq_ind]
            diff = (
                input_sequence[ba_ind, seq_ind]
                - final_encoder_hidden[ba_ind, :, xy[1], xy[0]]
            )
            assert diff.sum() == 0

    def extract_features(
        self,
        x: torch.Tensor,
        prev_output_tokens: torch.Tensor,
        final_encoder_hidden: torch.Tensor,
        encoder_feats: Optional[torch.Tensor] = None,
        projected_encoder_feats: Optional[torch.Tensor] = None,
        encoder_out: Optional[tuple] = None,
        incremental_state: Optional[dict] = None,
    ):
        """
        Similar to *forward* but only return features.
        from
        https://fairseq.readthedocs.io/en/latest/_modules/fairseq/models/lstm.html#LSTMDecoder
        """
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]
            encoder_padding_mask = encoder_out[3]
        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        srclen = encoder_outs.size(0)
        bsz, seqlen, _ = prev_output_tokens.size()

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # # initialize previous states (or get from cache during incremental generation)
        # if incremental_state is not None and len(incremental_state) > 0:
        #     prev_hiddens, prev_cells, input_feed = self.get_cached_state(
        #         incremental_state
        #     )
        if encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]

        input_feed = x.new_zeros(bsz, self.hidden_size)
        if self.input_feed_size == 0:
            input_feed = None

        sequence_mask = prev_output_tokens[:, :, 0] > 0

        outs = []

        # This remains the same as before.
        bsz, tgt_len, _ = prev_output_tokens.size()
        # bsz, hiden_dim, y, x
        # bsz, T, y, x
        xy_posi = (prev_output_tokens[:, :, :2] * 100).long().transpose(0, 1)
        xy_posi[xy_posi == -1] = 0

        encoder_hidden_dim = final_encoder_hidden.size(1)

        for j in range(seqlen):
            input_sequence = self.get_input_sequence(
                bsz, 1, xy_posi[j], final_encoder_hidden, encoder_hidden_dim
            )
            input = torch.cat((x[j, :, :], input_sequence), dim=-1)
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                # input = torch.cat((x[j, :, :], input_feed), dim=1)
                input = torch.cat((input, input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                # input = self.dropout_out_module(hidden)
                # if self.residuals:
                #     input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                att_res = self.attention(
                    h=hidden,
                    att_feats=encoder_feats,
                    p_att_feats=projected_encoder_feats,
                    att_masks=None,
                )
                out = torch.cat([hidden, att_res], dim=-1)
            else:
                out = hidden
            # out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                # attention is assumed
                out = self.after_attention_projection(out)
                input_feed = out

            # save final output
            outs.append(out)
            # from
            # https://github.com/husthuaan/AoANet/blob/master/models/AttModel.py#L145-L147
            # break if all the sequences end
            if j + 1 == seqlen:
                # collect outputs across time steps
                x = torch.cat(outs, dim=0).view(seqlen, bsz, -1)
                pad = None

            elif j >= 1 and sequence_mask[:, j + 1].sum() == 0:
                x = torch.cat(outs, dim=0).view(j + 1, bsz, -1)
                pad = (0, 0, 0, seqlen - (j + 1))
                break

        # Stack all the necessary tensors together and store
        # prev_hiddens_tensor = torch.stack(prev_hiddens)
        # prev_cells_tensor = torch.stack(prev_cells)
        # cache_state = torch.jit.annotate(
        #     Dict[str, Optional[Tensor]],
        #     {
        #         "prev_hiddens": prev_hiddens_tensor,
        #         "prev_cells": prev_cells_tensor,
        #         "input_feed": input_feed,
        #     },
        # )
        # self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        if pad is not None:
            x = torch.nn.functional.pad(x, pad=pad, mode="constant", value=0.0)

        # if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
        #     x = self.additional_fc(x)
        #     x = self.dropout_out_module(x)
        # # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        # if not self.training and self.need_attn and self.attention is not None:
        #     assert attn_scores is not None
        #     attn_scores = attn_scores.transpose(0, 2)
        # else:
        #     attn_scores = None
        return x, None


class Attention(nn.Module):
    def __init__(self, rnn_size: int = 128, att_hid_size: int = 128):
        """
        from
        https://github.com/husthuaan/AoANet/blob/master/models/AttModel.py
        """
        super(Attention, self).__init__()
        self.rnn_size = rnn_size
        self.att_hid_size = att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = nn.functional.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(
            -1, att_size, att_feats.size(-1)
        )  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(
            1
        )  # batch * att_feat_size

        return att_res
