"""
Hinton et al. (2015) Knowledge Distillation loss.

Total loss = alpha * KL(student_soft || teacher_soft) * T^2
           + (1 - alpha) * CrossEntropy(student_logits, hard_labels)

Why T^2?
  Softmax outputs shrink as temperature rises, making gradients smaller.
  Multiplying by T^2 re-scales the KL term so it stays on the same
  magnitude as the CE term regardless of temperature.

Why soft labels help (dark knowledge):
  Hard labels encode ONLY the correct class.
  Soft teacher labels encode inter-class similarity — e.g. a 'cat' image
  that also slightly resembles 'tiger' and 'dog'.  Training on these
  richer targets improves generalisation of the student.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Parameters
    ----------
    temperature : float
        Controls softness of probability distributions.
        T=1  → hard, peaked distributions (little dark knowledge transferred)
        T=4  → recommended default
        T=16 → very soft (works well when teacher is much larger than student)

    alpha : float  in [0, 1]
        Weight on the soft KL term.
        alpha=1.0 → ignore hard labels entirely (pure distillation)
        alpha=0.0 → ignore teacher (standard CE, no distillation)
        alpha=0.9 → recommended default
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.9):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        # # Experiment 1 - standard cross-entropy with hard labels
        # self.ce    = nn.CrossEntropyLoss()
        #Experiment 2 - label smoothing seems to help a lot when distilling with hard labels
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, student_logits, teacher_logits, labels):
        """
        student_logits : (B, C)  — raw output of the student network
        teacher_logits : (B, C)  — raw output of the teacher network (no_grad)
        labels         : (B,)    — integer hard class labels
        """

        # ── Soft-label KL divergence term ──────────────────────────────────────
        # log_softmax is numerically more stable than log(softmax(x))
        student_log_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft     = F.softmax(teacher_logits    / self.T, dim=1)

        # kl_div expects (input=log-probabilities, target=probabilities)
        # reduction='batchmean' gives the proper per-sample average
        kl_loss = F.kl_div(
            student_log_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.T ** 2)   # T^2 rescaling

        # ── Hard-label cross-entropy term ──────────────────────────────────────
        ce_loss = self.ce(student_logits, labels)

        # ── Weighted combination ───────────────────────────────────────────────
        total = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss

        return total, kl_loss.item(), ce_loss.item()


# ── Quick unit test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, C = 4, 100
    s_logits = torch.randn(B, C)
    t_logits = torch.randn(B, C)
    labels   = torch.randint(0, C, (B,))

    criterion = DistillationLoss(temperature=4.0, alpha=0.9)
    loss, kl, ce = criterion(s_logits, t_logits, labels)
    print(f"Total={loss:.4f}  KL={kl:.4f}  CE={ce:.4f}")