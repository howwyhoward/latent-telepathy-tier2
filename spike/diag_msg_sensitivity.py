"""How much does the trained policy's action actually move when the message flips?

Isaac-free. With a single neighbour the attention softmax over one unmasked slot
is exactly 1, so pooled = v(msg_proj(msg)) with NO dependence on the camera
image: the message's contribution to the actor's input is a closed-form
function of the wire. That makes the message's influence exactly measurable
without rolling out anything.

The number that decides the race is dmu[1] / sigma[1] -- how far the flipped bit
shifts the lateral action mean in units of the policy's own noise, since
P(top) = Phi(mu_y / sigma_y). Anything << 1 means the policy is ignoring the
message no matter what its weight norms look like.

    python spike/diag_msg_sensitivity.py --policy runs/race_v7/oracle.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chokepoint.constants import LATENT_DIM  # noqa: E402
from chokepoint.jepa import PixelEncoder  # noqa: E402
from chokepoint.receiver import AttentionReceiver  # noqa: E402

ANCHOR_DIMS = 2  # messages[:, :, 0:2] hold delta/comm_radius; broadcast follows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="runs/race_v7/oracle.pt")
    ap.add_argument("--wire", type=int, default=66)
    ap.add_argument("--samples", type=int, default=4096)
    args = ap.parse_args()

    sd = torch.load(args.policy, map_location="cpu")["policy"]
    # The encoder is never evaluated here; only the message half is exercised.
    policy = AttentionReceiver(
        PixelEncoder(LATENT_DIM),
        broadcast_dim=args.wire,  # receiver's msg_proj takes the full wire
        latent_dim=LATENT_DIM,
    )
    missing = policy.load_state_dict(sd, strict=False)
    assert not [k for k in missing.missing_keys if not k.startswith("encoder.")], \
        missing.missing_keys
    policy.eval()
    sigma = policy.log_std.exp()

    # Two wires identical but for the slab bit at index ANCHOR_DIMS.
    msg = torch.zeros(2, args.wire)
    msg[0, ANCHOR_DIMS] = 1.0    # slab top
    msg[1, ANCHOR_DIMS] = -1.0   # slab bottom

    with torch.no_grad():
        pooled = policy.v(policy.msg_proj(msg))
    d_pooled = pooled[0] - pooled[1]

    print(f"policy      {args.policy}")
    print(f"sigma       {[round(s, 3) for s in sigma.tolist()]}")
    print(f"||pooled||  top {pooled[0].norm():.4f}  bottom {pooled[1].norm():.4f}")
    print(f"||d_pooled|| {d_pooled.norm():.4f}   (message-induced shift in the "
          f"actor's 128-D message half)")

    # The ego half is unknown without images, so sweep it: the actor is a single
    # hidden layer, so a message effect that survives across random ego contexts
    # is a real effect and not an artefact of one operating point.
    d_mu = []
    with torch.no_grad():
        for scale in (0.0, 0.5, 1.0, 2.0):
            ego = torch.randn(args.samples, pooled.shape[-1]) * scale
            h_top = torch.cat([ego, pooled[0].expand_as(ego)], dim=-1)
            h_bot = torch.cat([ego, pooled[1].expand_as(ego)], dim=-1)
            d = policy.actor(h_top) - policy.actor(h_bot)
            d_mu.append((scale, d))
            print(f"  ego~N(0,{scale:.1f})  |d_mu| per dim "
                  f"{[round(x, 4) for x in d.abs().mean(0).tolist()]}"
                  f"   d_mu_y/sigma_y {float(d[:, 1].abs().mean() / sigma[1]):.4f}")

    best = max(float(d[:, 1].abs().mean() / sigma[1]) for _, d in d_mu)
    print(f"\nbest |d_mu_y| / sigma_y = {best:.4f}")
    print("P(top) shift from flipping the bit, at mu_y=0: "
          f"{0.5 - 0.5 * torch.erf(torch.tensor(-best / 2**0.5)).item():.4f} -> "
          "compare 0.50 for a policy that ignores the message entirely.")


if __name__ == "__main__":
    main()
