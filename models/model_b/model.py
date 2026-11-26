"""Encoder-only T5 grounding. No downloads occur on import."""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F

ATTRIBUTES = ('eyebrow_elevation', 'eye_openness', 'mouth_width',
              'smile_intensity', 'jaw_opening', 'cheek_raise')
SCHEMA_VERSION = '1.0'


def masked_mean(hidden, attention_mask):
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError('Expected hidden [B,T,H] and mask [B,T]')
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError('Attention mask must be binary')
    counts = attention_mask.sum(1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError('Empty token sequences are not valid')
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / counts.to(hidden.dtype)


class GroundingModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.d_model, 6)
        self.regressor = nn.Linear(encoder.config.d_model, 6)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               return_dict=True).last_hidden_state
        pooled = masked_mean(encoded, attention_mask)
        logits = self.classifier(pooled)
        return {'logits': logits, 'probabilities': logits.sigmoid(),
                'y': self.regressor(pooled).tanh()}

    @classmethod
    def from_pretrained(cls, model_id, revision, **kwargs):
        from transformers import T5EncoderModel
        if not revision:
            raise ValueError('An exact pretrained revision is required')
        return cls(T5EncoderModel.from_pretrained(model_id, revision=revision, **kwargs))


def supervised_loss(logits, predictions, labels, modified_mask, weights=None):
    """Attribute-mean BCE plus attribute-sum SSE, then example mean.

    Optional weights are dataset-normalized inverse group sizes with mean one.
    The fixed example denominator gives an unbiased group-equal minibatch loss;
    normalizing weights separately in each batch would distort that objective.
    """
    if logits.ndim != 2 or logits.shape[1] != 6 or any(
            tensor.shape != logits.shape for tensor in (predictions, labels, modified_mask)):
        raise ValueError('All loss tensors must have matching [B,6] shapes')
    if logits.shape[0] == 0 or not all(torch.isfinite(t).all() for t in
            (logits, predictions, labels, modified_mask)):
        raise ValueError('Loss inputs must be nonempty and finite')
    if not torch.all((modified_mask == 0) | (modified_mask == 1)):
        raise ValueError('Modified targets must be binary')
    values = F.binary_cross_entropy_with_logits(logits, modified_mask, reduction='none').mean(-1)
    values = values + (predictions - labels).square().sum(-1)
    if weights is not None:
        if weights.shape != values.shape or not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError('Weights must be positive finite [B] values')
        values = values * weights
    return values.mean()


def save_checkpoint(path, model, metadata, tokenizer=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(path / 'encoder_config')
    torch.save(model.state_dict(), path / 'weights.pt')
    payload = dict(metadata, attribute_order=list(ATTRIBUTES), schema_version=SCHEMA_VERSION)
    (path / 'metadata.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    if tokenizer is not None:
        tokenizer.save_pretrained(path / 'tokenizer')


def load_checkpoint(path, device='cpu'):
    from transformers import T5Config, T5EncoderModel
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['attribute_order'] != list(ATTRIBUTES) or metadata['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Checkpoint schema mismatch')
    model = GroundingModel(T5EncoderModel(T5Config.from_pretrained(path / 'encoder_config')))
    model.load_state_dict(torch.load(path / 'weights.pt', map_location=device, weights_only=True))
    return model.to(device).eval(), metadata


def tokenize_checked(tokenizer, texts, max_length, device='cpu'):
    if not texts or any(not isinstance(x, str) or not x.strip() for x in texts):
        raise ValueError('Nonempty instruction strings are required')
    tokens = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
    if tokens['input_ids'].shape[1] > max_length:
        raise ValueError('Instruction exceeds frozen max_length; silent truncation is forbidden')
    return {key: tokens[key].to(device) for key in ('input_ids', 'attention_mask')}


class Predictor:
    def __init__(self, model, tokenizer, metadata, device='cpu'):
        self.model, self.tokenizer, self.metadata = model.to(device).eval(), tokenizer, metadata
        self.device = device
        self.threshold = float(metadata['classification_threshold'])
        if not 0 <= self.threshold <= 1:
            raise ValueError('Invalid classification threshold')

    @torch.no_grad()
    def predict(self, texts):
        if not texts:
            return []
        output = self.model(**tokenize_checked(self.tokenizer, texts,
                            self.metadata['config']['max_length'], self.device))
        return [{'y': y, 'modified_mask': [int(p >= self.threshold) for p in probs],
                 'probabilities': probs, 'model_version': self.metadata['model_version'],
                 'schema_version': SCHEMA_VERSION}
                for y, probs in zip(output['y'].cpu().tolist(), output['probabilities'].cpu().tolist())]
