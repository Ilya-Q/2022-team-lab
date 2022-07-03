import torch
import torch.nn as nn

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
                #loss = second_score - first_score
                loss = self.loss(torch.stack([first_score, second_score]), torch.tensor([1., -1.]).to(self.model._target_device))
            elif label == 2:
                loss = self.loss(torch.stack([first_score, second_score]), torch.tensor([-1., 1.]).to(self.model._target_device))
                #loss = first_score - second_score

            total_loss += loss

            # Add terms that obs1 -> obs2 should be more consistent than obs2 -> obs1 for example (also with hypotheses)

            # Add terms that obs across things should generally have similar consistency

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
                #loss = second_score - first_score
                input, target = torch.stack([first_score, second_score]), torch.tensor([1., 0.]).to(self.model._target_device)
            elif label == 2:
                input, target = torch.stack([first_score, second_score]), torch.tensor([0., 1.]).to(self.model._target_device)
                #loss = first_score - second_score

                all_inputs.append(input)
                all_targets.append(target)

        input, target = torch.stack(all_inputs), torch.stack(all_targets)
        loss = self.loss(input, target)

        return loss