"""Train one shared policy for all robots by self-play (PPO with parameter sharing).

REFERENCE IMPLEMENTATION: the project's owner is building their own training pipeline as a
hands-on learning exercise (see HANDOFF.md). Study or compare against this file; it is not the plan.

    python scripts/train_selfplay.py --mode macro --timesteps 20000000 --name selfplay
    tensorboard --logdir runs

Worker processes each play full matches. In most of them all six robots are driven by the
current policy (self-play). In ``--vs-bots-frac`` of them one alliance is scripted bots,
which gives a steady progress curve: ``vs_bots/win`` and ``vs_bots/margin`` in TensorBoard.
Checkpoints (.pt) go to runs/<name>/; load them with watch.py / evaluate.py / play.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, replace
from multiprocessing import Pipe, Process
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy loads: one BLAS thread per worker

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from rebuilt_sim.env import EnvConfig, RewardConfig  # noqa: E402
from rebuilt_sim.obs import OBS_SIZE  # noqa: E402
from rebuilt_sim.policies import ActorCritic  # noqa: E402
from rebuilt_sim.pz_env import AGENTS, RebuiltParallelEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def worker(conn, cfg: EnvConfig, learners: tuple[str, ...], seed: int) -> None:
    try:
        torch.set_num_threads(1)
        env = RebuiltParallelEnv(cfg, learners=learners)
        episode = 0
        obs, _ = env.reset(seed=seed)
        conn.send(obs)
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            obs, rew, _, _, info = env.step(data)
            final = None
            if not env.agents:  # match over: report and start the next one
                final = info
                episode += 1
                obs, _ = env.reset(seed=seed + 7919 * episode)
            conn.send((obs, rew, final))
        env.close()
    except Exception:
        import traceback

        try:
            conn.send(("__error__", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()


def receive(conn, proc, timeout: float = 300.0):
    """Wait for a worker's reply; raise with its traceback if it failed or died."""
    waited = 0.0
    while not conn.poll(1.0):
        waited += 1.0
        if not proc.is_alive():
            raise RuntimeError(f"worker {proc.name} exited unexpectedly (exit code {proc.exitcode})")
        if waited > timeout:
            raise TimeoutError(f"worker {proc.name} sent nothing for {timeout:.0f} s")
    msg = conn.recv()
    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__error__":
        raise RuntimeError(f"worker {proc.name} failed:\n{msg[1]}")
    return msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["macro", "continuous"], default="macro")
    ap.add_argument("--name", default=None)
    ap.add_argument("--timesteps", type=int, default=20_000_000, help="total agent decisions (all robots)")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 4)))
    ap.add_argument("--vs-bots-frac", type=float, default=0.25)
    ap.add_argument("--tiers", nargs="+", default=["strong", "mid", "elite", "low"],
                    help="robot types the learners drive, sampled per robot per match")
    ap.add_argument("--rollout", type=int, default=256, help="decisions per robot per update")
    ap.add_argument("--decision-dt", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="a .pt checkpoint to continue from")
    ap.add_argument("--save-every", type=int, default=25, help="updates between checkpoints")
    args = ap.parse_args()

    macro = args.mode == "macro"
    decision_dt = args.decision_dt or (0.25 if macro else 0.1)
    gamma = args.gamma or (0.995 if macro else 0.997)
    ent_coef = args.ent_coef if args.ent_coef is not None else (0.01 if macro else 0.0)
    name = args.name or f"selfplay-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(action_mode=args.mode, decision_dt=decision_dt, learner_tiers=tuple(args.tiers),
                    reward=replace(RewardConfig()))
    (out / "config.json").write_text(json.dumps({"env": asdict(cfg), **vars(args), "gamma": gamma,
                                                 "ent_coef": ent_coef}, indent=2, default=str))
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    n_bot_workers = round(args.workers * args.vs_bots_frac)
    learners = []
    for w in range(args.workers):
        if w < n_bot_workers:
            learners.append(AGENTS[:3] if w % 2 == 0 else AGENTS[3:])
        else:
            learners.append(AGENTS)
    conns, procs = [], []
    for w in range(args.workers):
        a, b = Pipe()
        p = Process(target=worker, args=(b, cfg, learners[w], args.seed * 100_003 + w), daemon=True,
                    name=f"match-worker-{w}")
        p.start()
        b.close()  # the child owns this end; closing ours lets recv() see it if the child dies
        conns.append(a)
        procs.append(p)
    obs = [receive(c, p) for c, p in zip(conns, procs)]
    streams = [(w, a) for w in range(args.workers) for a in learners[w]]
    by_worker = [[(k, a) for k, (ww, a) in enumerate(streams) if ww == w] for w in range(args.workers)]
    S, T = len(streams), args.rollout
    act_dim = 8 if macro else 6

    net = ActorCritic.load(args.resume) if args.resume else ActorCritic(OBS_SIZE, act_dim, discrete=macro)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    writer = SummaryWriter(str(out))
    b_obs = torch.zeros(T, S, OBS_SIZE)
    b_act = torch.zeros(T, S, dtype=torch.long) if macro else torch.zeros(T, S, act_dim)
    b_logp, b_val, b_rew, b_done = (torch.zeros(T, S) for _ in range(4))
    updates = math.ceil(args.timesteps / (T * S))
    print(f"self-play {args.mode}: {args.workers} matches in parallel, {S} robots learning, "
          f"{updates} updates of {T * S:,} decisions -> {out}")
    step_count, t_start = 0, time.time()
    stats: dict[str, list[float]] = defaultdict(list)

    for update in range(1, updates + 1):
        for g in opt.param_groups:  # linear learning-rate decay
            g["lr"] = args.lr * (1.0 - (update - 1) / updates)
        for t in range(T):
            ob = torch.as_tensor(np.stack([obs[w][a] for w, a in streams]), dtype=torch.float32)
            act, logp, val = net.act(ob)
            b_obs[t], b_act[t], b_logp[t], b_val[t] = ob, act, logp, val
            per_worker = defaultdict(dict)
            acts = act.numpy()
            for k, (w, a) in enumerate(streams):
                per_worker[w][a] = int(acts[k]) if macro else acts[k]
            for w, c in enumerate(conns):
                c.send(("step", per_worker[w]))
            for w, c in enumerate(conns):
                o, rew, final = receive(c, procs[w])
                obs[w] = o
                for k, a in by_worker[w]:
                    b_rew[t, k] = rew[a]
                    b_done[t, k] = float(final is not None)
                if final is not None:
                    group = "vs_bots" if len(learners[w]) == 3 else "selfplay"
                    first = final[learners[w][0]]
                    stats[f"{group}/own_score"].append(first["own_score"])
                    stats[f"{group}/own_fuel"].append(first["own_fuel"])
                    stats[f"{group}/ranking_points"].append(first["ranking_points"])
                    if group == "vs_bots":
                        stats["vs_bots/win"].append(first["win"])
                        stats["vs_bots/margin"].append(first["own_score"] - first["opp_score"])
            step_count += S

        with torch.no_grad():
            last_val = net.value(torch.as_tensor(np.stack([obs[w][a] for w, a in streams]), dtype=torch.float32))
        adv = torch.zeros(T, S)
        gae = torch.zeros(S)
        for t in reversed(range(T)):
            nonterminal = 1.0 - b_done[t]
            next_val = last_val if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + gamma * next_val * nonterminal - b_val[t]
            gae = delta + gamma * args.lam * nonterminal * gae
            adv[t] = gae
        ret = adv + b_val

        f_obs, f_act = b_obs.reshape(T * S, -1), b_act.reshape(T * S, *b_act.shape[2:])
        f_logp, f_adv, f_ret = b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        n = T * S
        clipfracs, kls = [], []
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, args.minibatch):
                mb = perm[i:i + args.minibatch]
                new_logp, entropy, v = net.evaluate(f_obs[mb], f_act[mb])
                ratio = (new_logp - f_logp[mb]).exp()
                a_mb = f_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - args.clip, 1 + args.clip) * a_mb).mean()
                vloss = 0.5 * (v - f_ret[mb]).pow(2).mean()
                loss = pg + args.vf_coef * vloss - ent_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kls.append(((ratio - 1) - (new_logp - f_logp[mb])).mean().item())
                    clipfracs.append(((ratio - 1).abs() > args.clip).float().mean().item())

        var_y = f_ret.var()
        explained = float(1 - (f_ret - b_val.reshape(-1)).var() / var_y) if var_y > 0 else float("nan")
        sps = step_count / (time.time() - t_start)
        writer.add_scalar("train/policy_loss", pg.item(), step_count)
        writer.add_scalar("train/value_loss", vloss.item(), step_count)
        writer.add_scalar("train/entropy", entropy.mean().item(), step_count)
        writer.add_scalar("train/approx_kl", float(np.mean(kls)), step_count)
        writer.add_scalar("train/clip_fraction", float(np.mean(clipfracs)), step_count)
        writer.add_scalar("train/explained_variance", explained, step_count)
        writer.add_scalar("train/learning_rate", opt.param_groups[0]["lr"], step_count)
        writer.add_scalar("time/decisions_per_second", sps, step_count)
        line = f"update {update}/{updates}  decisions {step_count:,}  {sps:,.0f}/s"
        for k, v in sorted(stats.items()):
            writer.add_scalar(k, float(np.mean(v)), step_count)
            line += f"  {k} {np.mean(v):.2f}"
        print(line, flush=True)
        stats.clear()
        if update % args.save_every == 0 or update == updates:
            net.save(out / f"policy_{step_count}.pt", extra={"decisions": step_count, "mode": args.mode})
            net.save(out / "latest.pt", extra={"decisions": step_count, "mode": args.mode})

    for c in conns:
        c.send(("close", None))
    for p in procs:
        p.join(timeout=10)
    writer.close()
    print(f"saved {out / 'latest.pt'}")


if __name__ == "__main__":
    main()
