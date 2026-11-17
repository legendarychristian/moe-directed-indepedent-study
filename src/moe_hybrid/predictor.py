
import torch
import torch.nn.functional as F


class ExpertMapPredictor:
    def __init__(self, history_path):
        data = torch.load(
            history_path,
            map_location="cpu",
            weights_only=True,
        )

        self.history = (
            data["expert_maps"]
            .float()
            .reshape(-1, 16, 64)
        )

        self.prefixes = {
            n: F.normalize(
                self.history[:, :n].reshape(-1, n * 64),
                dim=-1,
            )
            for n in range(4, 16)
        }

        self.mean_top1 = (
            self.history.mean(dim=0)
            .argmax(dim=-1)
            .tolist()
        )

    def predict(self, observed, known, method="nn"):
        """
        Predict the highest-ranked expert in layer `known`
        using routing probabilities from preceding layers.

        Returns:
            (predicted_expert_id, cosine_similarity)
        """
        if method not in ("nn", "mean"):
            raise ValueError(f"Unknown predictor: {method}")

        query = F.normalize(
            torch.cat([
                observed[i] for i in range(known)
            ]),
            dim=0,
        )

        similarities = self.prefixes[known] @ query

        best_id = int(similarities.argmax())
        confidence = float(similarities[best_id])

        if method == "nn":
            expert_id = int(
                self.history[best_id, known].argmax()
            )
        else:
            expert_id = int(self.mean_top1[known])

        return expert_id, confidence
