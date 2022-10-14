import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, batch_first=True):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size     = hidden_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm            = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc              = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        out_emb     = self.word_embeddings(captions[:,:-1])
        out_lstm, _ = self.lstm(torch.cat((features.unsqueeze(1), out_emb), dim=1))
        out_linear  = self.fc(out_lstm)  
            
        return out_linear

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []

        for i in range(max_len):
            out_lstm, states = self.lstm(inputs, states)
            out_linear       = self.fc(out_lstm.squeeze(1))
            _, predicted     = out_linear.max(1)
            inputs           = self.word_embeddings(predicted).unsqueeze(1)
            
            item = predicted.item()
            if item == 1:
                break
            elif item != 0:
                result.append(item)
            
        return result