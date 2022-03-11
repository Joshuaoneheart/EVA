import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig
from pooling import SAP

"""### Define model"""


class PoolingHead(nn.Module):
    def __init__(self):
        super(PoolingHead, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.pooling = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x, mask=None):
        if(mask == None):
            return self.pooling(x)[:, 0, :]
        return self.pooling(x, src_key_padding_mask=mask.transpose(0, 1))[:, 0, :]


class AcousticEncoder(nn.Module):
    def __init__(self):
        super(AcousticEncoder, self).__init__()
        self.config = HubertConfig()
        self.hubert_layers = HubertModel(self.config)
        # self.hubert_layers.feature_extractor._freeze_parameters()

    def forward(self, x, mask):
        x = self.hubert_layers(input_values=x, attention_mask=mask)
        #print(f'x = {x}')
        return x


class DualEncoder(nn.Module):
    def __init__(self, device="cuda:0", T=0.1, K=1024):
        super(DualEncoder, self).__init__()
        self.acousticModel = AcousticEncoder()
        self.PoolingHead = SAP(out_dim=768)

        self.device = device
        self.T = T
        self.K = K
        self.register_buffer("queue", torch.zeros(
            768, self.K, dtype=torch.float))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        '''
        self.cls = nn.Sequential(
            nn.GELU(),
            nn.Linear(768, 7))
        '''

    @torch.no_grad()
    def _dequeue_and_enqueue(self, encoding):
        batch_size = encoding.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr: ptr + batch_size] = encoding.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, audio, audio_mask, text_encoding):
        audio_encoding = self.acousticModel(audio, audio_mask)
        audio_encoding = self.PoolingHead(audio_encoding.last_hidden_state, self.acousticModel.hubert_layers._get_feature_vector_attention_mask(
            audio_encoding.last_hidden_state.shape[1], audio_mask))
        l_pos = torch.einsum(
            'nc, nc->n', [audio_encoding, text_encoding]).unsqueeze(-1)

        l_neg = torch.einsum(
            'nc, ck->nk', [audio_encoding, self.queue.detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        #self.queue_length += config["train_batch_size"]
        return logits, labels, text_encoding

    def inference(self, audio, audio_mask):
        audio_encoding = self.acousticModel(audio, audio_mask)
        audio_encoding = self.PoolingHead(audio_encoding.last_hidden_state, self.acousticModel.hubert_layers._get_feature_vector_attention_mask(
            audio_encoding.last_hidden_state.shape[1], audio_mask))
        return self.cls(audio_encoding)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.GELU(),
            nn.Linear(768, 6)
        )

    def forward(self, x):
        return self.linear(x)
