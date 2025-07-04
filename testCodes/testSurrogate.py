import torch
from snntorch import surrogate

# pick a surrogate factory
surr_fn = surrogate.atan(alpha=2.0)

# make some “membrane potentials” around threshold
x = torch.linspace(-3, +3, steps=101, requires_grad=True)

# forward through the surrogate spike nonlinearity
y = surr_fn(x)    # y is 0/1 but has a custom backward

# take a simple scalar function of the spikes
loss = y.sum()

# backprop
loss.backward()

# plot or print the gradient d(loss)/d(x) = dS/dU
print(x.detach().cpu().numpy())
print(x.grad.detach().cpu().numpy())
