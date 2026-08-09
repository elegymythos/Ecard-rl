"""阶段 3：PPO 核心（约 200 行）。

设计（direction.md 阶段 3）：
- actor-critic 64 维、GAE、clip、熵正则；皇帝/奴隶两个独立 agent。
- 优势归一化默认关，做成 flag——旧 README 结果的元凶，要单独研究。
- 环境只给客观奖励，主观效用由 utility 层包装后进入各自 buffer。
- 单 seed 结果只记录，不下「收敛」结论。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from env import ECardEnv, PLAY_ACE

_ILLEGAL = -1e9


class Agent(nn.Module):
    """共享状态的 actor-critic：64→64，actor=2 个 logits，critic=标量值。"""

    def __init__(self, state_dim: int = 5, hidden: int = 64, n_actions: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.fc(obs)
        return self.actor_head(h), self.critic_head(h)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: torch.Tensor | None = None):
        """单步采样。返回 (action, logp, value, probs)。"""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(obs_t)
        if mask is not None:
            logits = logits + mask
        dist = Categorical(logits=logits)
        action = dist.sample()
        return (int(action.item()), dist.log_prob(action).item(),
                float(value.item()), dist.probs.squeeze(0).numpy())

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return float(self.critic_head(self.fc(obs_t)).item())


class Buffer:
    """单 agent 的 rollout buffer + GAE 计算。"""

    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.logps: list[float] = []
        self.values: list[float] = []
        self.next_values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def add(self, state, action, logp, value, next_value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logps.append(logp)
        self.values.append(value)
        self.next_values.append(next_value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)

    def get(self, gamma: float, lam: float, adv_norm: bool):
        """返回 (states, actions, old_logps, advantages, returns)。"""
        states = torch.as_tensor(np.stack(self.states), dtype=torch.float32)
        actions = torch.as_tensor(np.array(self.actions), dtype=torch.long)
        old_logps = torch.as_tensor(np.array(self.logps), dtype=torch.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            cont = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values[t] * cont - values[t]
            gae = delta + gamma * lam * cont * gae
            adv[t] = gae
        returns = adv + values

        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        if adv_norm:  # 默认关：它抹掉 +1/-5 的绝对尺度，是旧结果的可疑元凶
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return (states, actions, old_logps, adv_t,
                torch.as_tensor(returns, dtype=torch.float32))


def _mask(legal_actions: list[int]) -> torch.Tensor:
    m = np.full(2, _ILLEGAL, dtype=np.float32)  # 非法动作给大负数
    m[legal_actions] = 0.0
    return torch.as_tensor(m, dtype=torch.float32).unsqueeze(0)


def collect_rollout(env, agents, utilities, steps, gamma, lam):
    """跑 steps 步自博弈，返回 (buffers, stats)。

    utilities: {"emperor": u_e, "slave": u_s}，u(reward, context) -> 主观奖励。
    stats: 皇帝胜率、皇帝客观期望、双方主观奖励均值。
    """
    buffers = {"emperor": Buffer(), "slave": Buffer()}
    obs = env.reset()
    streak = {"emperor": 0, "slave": 0}
    ep_returns = {"emperor": 0.0, "slave": 0.0}
    subj_sums = {"emperor": 0.0, "slave": 0.0}
    obj_returns: list[float] = []
    wins = 0
    episodes = 0

    for _ in range(steps):
        mask_e = _mask(env.legal_actions("emperor"))
        mask_s = _mask(env.legal_actions("slave"))
        a_e, lp_e, v_e, _ = agents["emperor"].act(obs, mask_e)
        a_s, lp_s, v_s, _ = agents["slave"].act(obs, mask_s)
        next_obs, r_obj, done, info = env.step(a_e, a_s)

        # 心理层：皇帝看自己的客观奖励，奴隶看自己的（零和镜像）
        ctx_e = {"streak_losses": streak["emperor"]}
        ctx_s = {"streak_losses": streak["slave"]}
        r_e = utilities["emperor"](r_obj, ctx_e)
        r_s = utilities["slave"](-r_obj, ctx_s)

        nv_e = 0.0 if done else agents["emperor"].value(next_obs)
        nv_s = 0.0 if done else agents["slave"].value(next_obs)
        buffers["emperor"].add(obs, a_e, lp_e, v_e, nv_e, r_e, done)
        buffers["slave"].add(obs, a_s, lp_s, v_s, nv_s, r_s, done)

        subj_sums["emperor"] += r_e
        subj_sums["slave"] += r_s
        ep_returns["emperor"] += r_obj
        ep_returns["slave"] += -r_obj
        # 连败计数：平局不算连败（先按本回合前的 streak 给 context，再更新）
        streak["emperor"] = 0 if r_obj >= 0 else streak["emperor"] + 1
        streak["slave"] = 0 if r_obj <= 0 else streak["slave"] + 1

        if done:
            wins += int(info["winner"] == "emperor")
            episodes += 1
            obj_returns.append(ep_returns["emperor"])
            ep_returns = {"emperor": 0.0, "slave": 0.0}
            streak = {"emperor": 0, "slave": 0}
            obs = env.reset()
        else:
            obs = next_obs

    stats = {
        "emperor_win_rate": wins / episodes if episodes else float("nan"),
        "emperor_obj_return": float(np.mean(obj_returns)) if obj_returns else float("nan"),
        "subj_mean": {k: subj_sums[k] / steps for k in subj_sums},
    }
    return buffers, stats


def update_policy(agent, optimizer, buffer, *, gamma, lam, clip_eps, ent_coef,
                  adv_norm, epochs, batch_size):
    """对单个 agent 做 PPO 更新。返回 (policy_loss, value_loss, entropy)。"""
    states, actions, old_logps, adv, returns = buffer.get(gamma, lam, adv_norm)
    n = len(states)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, values = agent(states[idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])
            ratio = (logp - old_logps[idx]).exp()
            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()
            value_loss = F.mse_loss(values.squeeze(-1), returns[idx])
            loss = policy_loss + 0.5 * value_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return policy_loss.item(), value_loss.item(), entropy.item()


@torch.no_grad()
def first_round_probs(agents):
    """初始状态下皇帝/奴隶出王牌的概率（阶段 3 验证门的主角 p̂、q̂）。"""
    obs = ECardEnv().reset()
    mask = _mask([0, 1])
    _, _, _, probs_e = agents["emperor"].act(obs, mask)
    _, _, _, probs_s = agents["slave"].act(obs, mask)
    return float(probs_e[PLAY_ACE]), float(probs_s[PLAY_ACE])
