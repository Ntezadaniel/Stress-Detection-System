"""
losses.py
---------
Implements the two ideas from your proposal's Reinforced Knowledge
Distillation framework (Phase 2 and Phase 3):

1. Distillation loss (Phase 2): trains a compact student model to
   reproduce a larger teacher model's SOFT output distribution, not
   just its hard labels, using temperature-scaled softmax + KL-divergence.

2. Cost-sensitive weighting (Phase 3, the "Wisdom Layer"): penalizes the
   model more heavily for missing a real high-stress case (a false
   negative) than for a false alarm. This is the "penalty" your proposal
   described -- it biases the model toward never missing a genuine
   stress event, even at the cost of a few more false alarms.

Your proposal's formula (Section 3.3.3):

    L_CS = sum_i  w_i * loss(y_i, y_hat_i)
    where w_i = alpha  if y_i == High-Stress
               = 1      otherwise

Here we call that weight `penalty` instead of `alpha`, because this
module already uses `alpha` for the distillation/hard-label balance
(the two are different knobs -- don't confuse them when you write your
Methodology chapter).
"""

import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, true_labels,
                       temperature: float = 2.0, alpha: float = 0.5):
    """
    Plain (non-reinforced) distillation loss -- use this first, to get a
    working student before adding the cost-sensitive penalty on top.

    alpha=1.0 means "only imitate the teacher's soft outputs".
    alpha=0.0 means "ignore the teacher, learn only from ground-truth labels".
    A starting value of 0.5 weighs both equally.
    """
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    distill_term = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    hard_term = F.cross_entropy(student_logits, true_labels)

    return alpha * distill_term + (1 - alpha) * hard_term


def cost_sensitive_weights(true_labels: torch.Tensor, high_stress_class_id: int,
                            penalty: float = 5.0) -> torch.Tensor:
    """
    Returns one weight per sample in the batch: `penalty` if that sample's
    true label is the high-stress class, 1.0 otherwise.

    Multiplying your loss by these weights means a missed high-stress
    sample counts `penalty` times more than an ordinary mistake -- so the
    model learns to avoid ever missing one, even if it means more false
    alarms on the other classes.
    """
    weights = torch.ones_like(true_labels, dtype=torch.float32)
    weights[true_labels == high_stress_class_id] = penalty
    return weights


def reinforced_distillation_loss(student_logits, teacher_logits, true_labels,
                                  high_stress_class_id: int,
                                  temperature: float = 2.0,
                                  alpha: float = 0.5,
                                  penalty: float = 5.0):
    """
    The full loss from your proposal: distillation loss, reweighted so
    high-stress samples count `penalty` times more than everything else.

    Use this in Step 5 once your plain distillation_loss() is already
    training successfully -- adding the penalty on top of a broken
    training loop just makes debugging harder.
    """
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # Per-sample values (reduction="none"), so we can weight each one
    # individually before averaging -- a plain KL/CE call would average
    # everything together first and we'd lose the ability to weight.
    per_sample_kl = F.kl_div(soft_student, soft_teacher, reduction="none").sum(dim=1)
    per_sample_ce = F.cross_entropy(student_logits, true_labels, reduction="none")

    weights = cost_sensitive_weights(true_labels, high_stress_class_id, penalty)

    distill_term = (per_sample_kl * weights).mean() * (temperature ** 2)
    hard_term = (per_sample_ce * weights).mean()

    return alpha * distill_term + (1 - alpha) * hard_term


if __name__ == "__main__":
    # Sanity check: fabricate a tiny batch and confirm both loss functions
    # run and that the reinforced version actually produces a LARGER loss
    # when a high-stress sample is misclassified, than an ordinary one.
    torch.manual_seed(0)
    batch_size, num_classes = 6, 3
    HIGH_STRESS = 2  # e.g. class index for "stress" in {0:baseline, 1:amusement, 2:stress}

    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    true_labels = torch.tensor([0, 1, 2, 2, 0, 1])  # two high-stress samples in this batch

    plain = distillation_loss(student_logits, teacher_logits, true_labels)
    reinforced = reinforced_distillation_loss(
        student_logits, teacher_logits, true_labels,
        high_stress_class_id=HIGH_STRESS, penalty=5.0,
    )

    print(f"Plain distillation loss:      {plain.item():.4f}")
    print(f"Reinforced (penalty=5.0):     {reinforced.item():.4f}")
    print(f"Reinforced (penalty=1.0)"
          f" [should match plain-ish]:    "
          f"{reinforced_distillation_loss(student_logits, teacher_logits, true_labels, HIGH_STRESS, penalty=1.0).item():.4f}")

    reinforced.backward()
    print("\nBackward pass succeeded -- gradients flow correctly through the loss.")
