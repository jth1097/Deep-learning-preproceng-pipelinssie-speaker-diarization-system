# model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

class Wav2VecWithClassifier(nn.Module):
    def __init__(self, hidden_dim=1024, num_labels=2, layer=14):
        super().__init__()
        
        # Load the pre-trained wav2vec model
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust")
        self.wav2vec.freeze_feature_extractor()
        self.classifier = nn.Linear(hidden_dim, num_labels)
  
        self.layer = layer
        self.hidden_dim = hidden_dim
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust")

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    

    def forward(self, input_values):
        # Convert inputs to tensor if needed
        inputs = self.feature_extractor(
            input_values,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
        )["input_values"][0]

        # Move inputs to device
        inputs = inputs.to(self.wav2vec.device)

        # Get wav2vec outputs with hidden states
        outputs = self.wav2vec(
            input_values=inputs, 
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[self.layer + 1]

        # Apply classifier
        logits = self.classifier(hidden_states)

        return {"logits": logits}
