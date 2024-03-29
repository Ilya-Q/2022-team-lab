import torch
import torch.nn as nn

# Author: Ilya Kuryanov & Patrick Bareiß
class MSELoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()

    def forward(self, sentence_features, labels):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        total_loss = torch.tensor(0., requires_grad=True).to(self.model._target_device)
        for i, label in enumerate(labels):
            obs1, obs2, hyp1, hyp2 = embeddings[0][i], embeddings[1][i], embeddings[2][i], embeddings[3][i]
            first_score = self.model.consistent(self.model.occurs_after(self.model.occurs_after(obs1, hyp1), obs2))
            second_score = self.model.consistent(self.model.occurs_after(self.model.occurs_after(obs1, hyp2), obs2))

            if label == 1:
                loss = self.loss(torch.stack([first_score, second_score]), torch.tensor([1., -1.]).to(self.model._target_device))
            elif label == 2:
                loss = self.loss(torch.stack([first_score, second_score]), torch.tensor([-1., 1.]).to(self.model._target_device))

            total_loss += loss

        return total_loss


class CELoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features, labels):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        all_inputs = []
        all_targets = []
        for i, label in enumerate(labels):
            obs1, obs2, hyp1, hyp2 = embeddings[0][i], embeddings[1][i], embeddings[2][i], embeddings[3][i]
            first_score = self.model.consistent(self.model.occurs_after(self.model.occurs_after(obs1, hyp1), obs2))
            second_score = self.model.consistent(self.model.occurs_after(self.model.occurs_after(obs1, hyp2), obs2))

            if label == 1:
                input, target = torch.stack([first_score, second_score]), torch.tensor([1., 0.]).to(self.model._target_device)
            elif label == 2:
                input, target = torch.stack([first_score, second_score]), torch.tensor([0., 1.]).to(self.model._target_device)

                all_inputs.append(input)
                all_targets.append(target)

        if len(all_inputs) == 0 or len(all_targets) == 0:
            return torch.tensor(0., requires_grad=True).cuda()
        input, target = torch.stack(all_inputs), torch.stack(all_targets)
        loss = self.loss(input, target)

        return loss