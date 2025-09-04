import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F


class TopkRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        logits = self.linear(x)

        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices


class TextModel(nn.Module):
    def __init__(self, hyp_params):
        super(TextModel, self).__init__()
        self.orig_l_len = hyp_params.l_len
        self.orig_d_l = hyp_params.orig_d_l
        self.embed_dim = hyp_params.embed_dim
        self.out_dropout = hyp_params.out_dropout
        self.language = hyp_params.language
        self.finetune = hyp_params.finetune

        self.output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=1, bias=False)

        # Prepare BERT model
        self.text_model = BertTextEncoder(language=hyp_params.language)

        # Unimodal encoder
        self.encoder_l = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)

        # Projection layers
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, x_l):
        x_l = self.text_model(x_l, use_finetune=self.finetune)
        #################################################################################
        # Project the textual features
        x_l = x_l.transpose(1, 2)
        proj_x_l = self.proj_l(x_l)  # (bs, embed, seq)
        proj_x_l = proj_x_l.permute(2, 0, 1)  # (seq, bs, embed)
        #################################################################################
        # Unimodal encoder
        h_l = self.encoder_l(proj_x_l)
        #################################################################################
        # Predict
        h_l = torch.mean(h_l, dim=0)  # (bs, embed)
        h_l = F.normalize(h_l, p=2, dim=-1)
        last_hs = h_l
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        final_output = self.out_layer(last_hs_proj)
        outputs = {
            'pred': final_output,
            'last_hs_proj': last_hs_proj,
            'h_l': h_l
        }

        return outputs


class AudioModel(nn.Module):
    def __init__(self, hyp_params):
        super(AudioModel, self).__init__()
        self.orig_a_len = hyp_params.a_len
        self.orig_d_a = hyp_params.orig_d_a
        self.embed_dim = hyp_params.embed_dim
        self.out_dropout = hyp_params.out_dropout

        self.output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)
        self.norm_a = nn.BatchNorm1d(self.orig_d_a)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=1, bias=False)

        # Unimodal encoder
        self.encoder_a = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)

        # Projection layers
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, x_a):
        #################################################################################
        # Project the audio features
        x_a = x_a.transpose(1, 2)
        x_a = self.norm_a(x_a)
        proj_x_a = self.proj_a(x_a)  # (bs, embed, seq)
        proj_x_a = proj_x_a.permute(2, 0, 1)  # (seq, bs, embed)
        #################################################################################
        # Unimodal encoder
        h_a = self.encoder_a(proj_x_a)
        #################################################################################
        # Predict
        h_a = torch.mean(h_a, dim=0)  # (bs, embed)
        h_a = F.normalize(h_a, p=2, dim=-1)
        last_hs = h_a
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        final_output = self.out_layer(last_hs_proj)
        outputs = {
            'pred': final_output,
            'last_hs_proj': last_hs_proj,
            'h_a': h_a
        }

        return outputs


class VisionModel(nn.Module):
    def __init__(self, hyp_params):
        super(VisionModel, self).__init__()
        self.orig_v_len = hyp_params.v_len
        self.orig_d_v = hyp_params.orig_d_v
        self.embed_dim = hyp_params.embed_dim
        self.out_dropout = hyp_params.out_dropout

        self.output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)
        self.norm_v = nn.BatchNorm1d(self.orig_d_v)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=1, bias=False)

        # Unimodal encoder
        self.encoder_v = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)

        # Projection layers
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, x_v):
        #################################################################################
        # Project the visual features
        x_v = x_v.transpose(1, 2)
        x_v = self.norm_v(x_v)
        proj_x_v = self.proj_v(x_v)  # (bs, embed, seq)
        proj_x_v = proj_x_v.permute(2, 0, 1)  # (seq, bs, embed)
        #################################################################################
        # Unimodal encoder
        h_v = self.encoder_v(proj_x_v)
        #################################################################################
        # Predict
        h_v = torch.mean(h_v, dim=0)  # (bs, embed)
        h_v = F.normalize(h_v, p=2, dim=-1)
        last_hs = h_v
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        final_output = self.out_layer(last_hs_proj)
        outputs = {
            'pred': final_output,
            'last_hs_proj': last_hs_proj,
            'h_v': h_v
        }

        return outputs


class CMCModel(nn.Module):
    def __init__(self, hyp_params):
        super(CMCModel, self).__init__()
        self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.embed_dim = hyp_params.embed_dim
        self.top_k = hyp_params.top_k
        self.out_dropout = hyp_params.out_dropout
        self.temperature = hyp_params.temperature
        self.language = hyp_params.language
        self.finetune = hyp_params.finetune

        self.output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)
        self.norm_a = nn.BatchNorm1d(self.orig_d_a)
        self.norm_v = nn.BatchNorm1d(self.orig_d_v)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=1, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=1, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=1, bias=False)

        # Prepare BERT model
        self.text_model = BertTextEncoder(language=hyp_params.language)

        # Unimodal encoder
        self.encoder_l = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)
        self.encoder_a = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)
        self.encoder_v = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)

        # MCR
        self.router = TopkRouter(embed_dim=self.embed_dim, num_experts=3, top_k=self.top_k)

        # Projection layers
        self.proj1s = nn.ModuleList([])
        self.proj2s = nn.ModuleList([])
        self.out_layers = nn.ModuleList([])
        for _ in range(3):  # (t, a, v) expert
            self.proj1s.append(nn.Linear(self.embed_dim, self.embed_dim))  # proj1s.x.weight / proj1s.x.bias
            self.proj2s.append(nn.Linear(self.embed_dim, self.embed_dim))  # proj2s.x.weight / proj2s.x.bias
            self.out_layers.append(
                nn.Linear(self.embed_dim, self.output_dim))  # out_layers.x.weight / out_layers.x.bias

    def forward(self, x_l, x_a, x_v):
        x_l = self.text_model(x_l, use_finetune=self.finetune)
        #################################################################################
        # Project the textual/audio/visual features
        x_l = x_l.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        x_a = self.norm_a(x_a)
        x_v = self.norm_v(x_v)
        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        proj_x_v = self.proj_v(x_v)  # (bs, embed, seq)
        proj_x_l = proj_x_l.permute(2, 0, 1)  # (seq, bs, embed)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        #################################################################################
        # Unimodal encoder
        h_l = self.encoder_l(proj_x_l)
        h_a = self.encoder_a(proj_x_a)
        h_v = self.encoder_v(proj_x_v)
        h_l = torch.mean(h_l, dim=0)  # (bs, embed)
        h_a = torch.mean(h_a, dim=0)  # (bs, embed)
        h_v = torch.mean(h_v, dim=0)  # (bs, embed)
        h_l = F.normalize(h_l, p=2, dim=-1)
        h_a = F.normalize(h_a, p=2, dim=-1)
        h_v = F.normalize(h_v, p=2, dim=-1)
        #################################################################################
        # Multimodal fusion
        h_ll = h_l.unsqueeze(1)  # (bs, 1, embed)
        h_aa = h_a.unsqueeze(1)
        h_vv = h_v.unsqueeze(1)
        x = torch.cat((h_ll, h_aa, h_vv), dim=1)  # (bs, 3, embed)

        # PFM
        cosine_similarity = torch.bmm(x, x.transpose(1, 2))  # (bs, 3, 3)
        sim_scores = F.softmax(cosine_similarity / self.temperature, dim=-1)
        x = torch.bmm(sim_scores, x)  # (bs, 3, embed)

        # MCR
        fused_x = torch.sum(x, dim=1)
        gating_output, indices = self.router(fused_x)  # gating_output - (bs, 3), indices - (bs, 1)

        bs = x.shape[0]
        final_output = torch.zeros(bs, self.output_dim).to(x.device)
        uni_outputs = [torch.zeros(bs, self.output_dim).to(x.device),
                       torch.zeros(bs, self.output_dim).to(x.device),
                       torch.zeros(bs, self.output_dim).to(x.device)]
        for i in range(3):
            expert_mask = (indices == i).any(dim=-1)  # chose which sample to use
            if expert_mask.any():
                last_hs = x[expert_mask][:, i]  # (select_bs, embed_dim)
                output_weights = gating_output[expert_mask][:, [i]]  # (select_bs, 1)
                last_hs_proj = self.proj2s[i](
                    F.dropout(F.relu(self.proj1s[i](last_hs)), p=self.out_dropout, training=self.training))
                last_hs_proj += last_hs
                output = self.out_layers[i](last_hs_proj)
                uni_outputs[i][expert_mask] += output
                output_weights = output_weights.expand(-1, self.output_dim)
                weighted_output = output_weights * output
                final_output[expert_mask] += weighted_output

        outputs = {
            'pred': final_output,
            'pred_t': uni_outputs[0],
            'pred_a': uni_outputs[1],
            'pred_v': uni_outputs[2],
            'h_l': h_l,
            'h_a': h_a,
            'h_v': h_v
        }

        return outputs


class EMA:
    def __init__(self, momentum=0.999):
        self.momentum = momentum
        self.global_grad = None

    def update(self, cur_global_grad):
        if self.global_grad is None:
            self.global_grad = cur_global_grad
        else:
            self.global_grad = self.momentum * self.global_grad + (1 - self.momentum) * cur_global_grad

    def step(self, new_momentum):
        self.new_momentum = new_momentum


class BertTextEncoder(nn.Module):
    def __init__(self, language='en'):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('/home/zhonggw/Pretrained_models/bert_en',
                                                             do_lower_case=True)
            self.model = model_class.from_pretrained('/home/zhonggw/Pretrained_models/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('/home/zhonggw/Pretrained_models/bert_cn')
            self.model = model_class.from_pretrained('/home/zhonggw/Pretrained_models/bert_cn')

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, text, use_finetune):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, features, labels):
        """
        features: [num_modals, feature_dim]
        labels: [num_modals]
        """

        similarity_matrix = torch.matmul(features, features.T)  # [num_modals, num_modals]

        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float()  # [num_modals, num_modals]
        mask_positive.fill_diagonal_(0)

        exp_sim = torch.exp(similarity_matrix / self.temperature)
        numerator = torch.sum(exp_sim * mask_positive, dim=1)
        denominator = torch.sum(exp_sim, dim=1) - exp_sim.diag()

        valid_pairs = mask_positive.sum(dim=1)
        valid_mask = valid_pairs > 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        ratio = numerator[valid_mask] / denominator[valid_mask]

        if self.reduction == 'mean':
            loss = -torch.log(ratio).mean()
        else:
            loss = -torch.log(ratio)

        return loss