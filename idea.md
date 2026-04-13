## Confidence-Balance Matrix Module (short)

### Core idea
Add an auxiliary matrix M in logit space that optimizes the model's confidence structure, not only top-1 accuracy. M should encourage a balanced view where non-target labels are treated as meaningful alternatives near hard boundaries.

### Why this makes sense
- Overconfident predictions usually fail on boundary cases.
- A confidence matrix can explicitly shape how probability mass is distributed across labels.
- This turns confusion into a training signal instead of a training error.

### Module behavior
For logits p (after softmax), apply a learnable adjustment:
q = softmax(M p)

Train with:
L = CE(y, q) + lambda * balance_loss(q)

balance_loss can penalize extreme confidence collapse and reward calibrated uncertainty on hard samples.

### Sample generation (image level)
Create hard samples directly in image space with gamma-controlled strength g:
1. Masking mix: x' = g * mask(x) + (1 - g) * x
2. Region concatenation: x' = concat(x_a, x_b, g)
3. Optional intensity gamma: x' = x^g

Small g keeps samples close to original; larger g pushes ambiguity.

### Training flow
1. Predict on clean x and sampled x'.
2. Use clean loss for correctness.
3. Use matrix-based confidence loss on sampled x' to learn balanced label confidence.

### One-line thesis
Learn a confidence-balance matrix with gamma-controlled hard sampling so the model stays accurate but less overconfident at class boundaries.


