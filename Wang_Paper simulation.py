"""
Yu Wang (2017) ACC - Simulation 1 / Figure 5
RNN vs Convex-based LSTM for nonlinear dynamic system identification.

NOTE ON EQ. (19):
The printed equation uses exp(y(k-1)) and exp(y(k-1)^2) with POSITIVE
exponents. As confirmed in the previous conversation, this causes
divergence at step k=5 (y = 2.5e+124). The entire reference chain
in this paper (Narendra & Parthasarathy 1990 [9], Wang 2014 [12],
Wang 2016 [11]) uses exp(-y^2) Gaussian kernels as the standard
bounded benchmark nonlinearity. The corrected stable form is:

    y(k) = 0.7*y(k-1) - 0.8*y(k-1)*exp(-y(k-1)^2)
           - 0.6*y(k-2) - 0.5*y(k-2)*exp(-y(k-2)^2)
           + u(k-1) + 0.3*u(k-2)

Everything else is implemented exactly as written in the paper:
  Eq.(8)  : LSTM gated unit equations
  Eq.(11) : Series-parallel identification structure
  Eq.(12) : Convex output
  Eq.(13) : Error definition
  Eq.(14) : Error decomposition
  Eq.(15) : Compact error form
  Eq.(16) : Alpha update law
  Eq.(17) : Convex error combination
  Eq.(18-19): System structure and input signal

HOW TO RUN:
    python wang2017_simulation.py

To push results to GitHub automatically after running:
    python wang2017_simulation.py --push

To specify your own repository remote URL:
    python wang2017_simulation.py --push --remote https://github.com/YOUR_USERNAME/YOUR_REPO.git

Requirements:
    pip install torch numpy matplotlib
    Git must be installed and configured on your machine.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# reproducibility
torch.manual_seed(0)
np.random.seed(0)

# -----------------------------------------------------------------------
# Simulation parameters (Section VII-A)
# -----------------------------------------------------------------------
T         = 2000   # time steps (x-axis of Figure 5)
n_mem     = 2      # output memory depth  (n=2, Eq. 18: y(k-1), y(k-2))
m_mem     = 2      # input  memory depth  (m=2, Eq. 18: u(k-1), u(k-2))
INPUT_DIM = 4      # [y(k-1), y(k-2), u(k-1), u(k-2)]
HIDDEN    = 15     # hidden units per network
LR        = 0.005  # learning rate for Adam
N         = 2      # number of LSTM sub-models in the convex cluster
SMOOTH    = 30     # moving-average window for the error plot


# -----------------------------------------------------------------------
# Section VII-A  --  Plant (Eq. 18-19, corrected exponent sign)
# -----------------------------------------------------------------------
def plant(y1, y2, u1, u2):
    # Eq.(19) with exp(-y^2) instead of exp(y) for stability
    return (0.7*y1 - 0.8*y1*np.exp(-y1**2)
            - 0.6*y2 - 0.5*y2*np.exp(-y2**2)
            + u1 + 0.3*u2)


def simulate_plant():
    k = np.arange(T)
    # Input signal from Section VII-A
    u = np.sin(2*np.pi*k/125) + np.cos(2*np.pi*k/50)
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = plant(y[t-1], y[t-2], u[t-1], u[t-2])
    return y, u


# -----------------------------------------------------------------------
# Section III-A  --  Conventional RNN
# -----------------------------------------------------------------------
class RNNModel(nn.Module):
    """
    Standard RNN as described in Section III-A.
    y(k) depends on input x(k) and fed-back state h(k-1).
    Single hidden layer, tanh activation, linear output.
    """
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(INPUT_DIM, HIDDEN, num_layers=1,
                          nonlinearity="tanh", batch_first=True)
        self.fc  = nn.Linear(HIDDEN, 1)

    def forward(self, x, h):
        out, h_new = self.rnn(x, h)
        return self.fc(out[:, -1, :]), h_new

    def init_hidden(self):
        return torch.zeros(1, 1, HIDDEN)


def run_rnn(y, u):
    """
    Online series-parallel identification (Eq. 11).
    Real past plant outputs y(k-1), y(k-2) are used as inputs
    instead of estimated outputs, ensuring stability.
    Error: e(k) = y(k) - y_hat(k)  (Eq. 13)
    """
    model = RNNModel()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    h     = model.init_hidden()
    err   = np.zeros(T)

    for t in range(2, T):
        x = torch.tensor(
            [[y[t-1], y[t-2], u[t-1], u[t-2]]], dtype=torch.float32
        ).unsqueeze(0)                                  # shape (1,1,4)
        yt = torch.tensor([[y[t]]], dtype=torch.float32)

        yhat, hn = model(x, h.detach())
        loss = (yt - yhat) ** 2                        # MSE loss, Eq.(13)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        err[t] = abs(y[t] - yhat.item())
        h = hn.detach()

    return err


# -----------------------------------------------------------------------
# Section III-C  --  Single LSTM sub-model (Eq. 8)
# -----------------------------------------------------------------------
class LSTMModel(nn.Module):
    """
    LSTM with three gated units as in Eq.(8):
        f(k) = sigmoid( W_f [h(k-1), x(k)] + b_f )   forget gate
        i(k) = sigmoid( W_i [h(k-1), x(k)] + b_i )   input  gate
        o(k) = sigmoid( W_o [h(k-1), x(k)] + b_o )   output gate
        C_tilde(k) = tanh( W_c [h(k-1), x(k)] + b_c )
        C(k) = f(k)*C(k-1) + i(k)*C_tilde(k)         memory cell
    PyTorch nn.LSTM implements exactly these equations.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(HIDDEN, 1)

    def forward(self, x, state):
        out, state_new = self.lstm(x, state)
        return self.fc(out[:, -1, :]), state_new

    def init_state(self):
        return (torch.zeros(1, 1, HIDDEN), torch.zeros(1, 1, HIDDEN))


# -----------------------------------------------------------------------
# Sections IV & VI  --  Convex-based LSTM Cluster (Eq. 12-17)
# -----------------------------------------------------------------------
class ConvexLSTMCluster:
    """
    N LSTM sub-models N_i combined by convex coefficients alpha_i.

    Convex output (Eq. 12):
        y_hat(k) = sum_i  alpha_i(k) * y_hat_i(k)

    Individual errors (Eq. 13, 14):
        e_i(k)       = y(k) - y_hat_i(k)
        e_tilde_i(k) = e_i(k) - e_n(k)

    Compact error form (Eq. 15):
        e_tilde(k) = E_tilde(k)^T * alpha_tilde(k)
        E_tilde = [e_tilde_1, ..., e_tilde_{n-1}]   row vector in R^(n-1)
        alpha_tilde = [alpha_1, ..., alpha_{n-1}]   (alpha_n = 1 - sum)

    Alpha update law (Eq. 16):
        alpha_tilde(k) = alpha_tilde(k-1)
                       - E_tilde(k-1)*E_tilde(k-1)^T * alpha_tilde(k-1)
                       + E_tilde(k-1) * e_tilde(k-1)

    Convergence property (Section VI-A):
        alpha error decreases exponentially -> identification converges
        even if individual BPTT has not converged (faster than RNN).

    Network weights updated simultaneously by standard backprop on e_i^2.
    """

    def __init__(self):
        self.n       = N
        self.models  = [LSTMModel() for _ in range(N)]
        self.opts    = [torch.optim.Adam(m.parameters(), lr=LR)
                        for m in self.models]
        self.states  = [m.init_state() for m in self.models]

        # alpha_tilde: n-1 free params; alpha_n = 1 - sum(alpha_tilde)
        # Initialised to 1/n so all models start with equal weight
        self.alpha_t = np.full(N - 1, 1.0 / N, dtype=np.float64)

        # Previous-step E_tilde and e_tilde needed by Eq.(16)
        self.E_prev  = np.zeros(N - 1, dtype=np.float64)
        self.e_prev  = 0.0

    def _full_alpha(self):
        """Full alpha vector satisfying convex constraint (Property 1)."""
        a = np.append(self.alpha_t, 1.0 - self.alpha_t.sum())
        a = np.clip(a, 0.0, 1.0)
        return a / a.sum()

    def step(self, x_np, y_val):
        """
        One online identification step.
        Returns |y(k) - y_hat_convex(k)|.
        """
        x  = torch.tensor(x_np, dtype=torch.float32).view(1, 1, INPUT_DIM)
        yt = torch.tensor([[y_val]], dtype=torch.float32)
        alpha = self._full_alpha()

        # Forward pass for all sub-models
        yhats, new_states = [], []
        for i, m in enumerate(self.models):
            yh, st = m(x, self.states[i])
            yhats.append(yh)
            new_states.append(st)

        # Convex output  Eq.(12)
        y_conv = sum(alpha[i] * yhats[i] for i in range(self.n))
        err    = abs(y_val - y_conv.item())

        # Per-model weight update on e_i^2  (Section VI, simultaneous)
        e_vals = []
        for i, (m, opt) in enumerate(zip(self.models, self.opts)):
            loss_i = (yt - yhats[i]) ** 2
            opt.zero_grad()
            loss_i.backward(retain_graph=(i < self.n - 1))
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            e_vals.append(y_val - yhats[i].item())   # e_i(k)  Eq.(13)

        # Alpha update  Eq.(16)
        e_n    = e_vals[-1]
        E_now  = np.array([e_vals[i] - e_n for i in range(self.n - 1)],
                          dtype=np.float64)            # E_tilde(k), Eq.(14)
        e_now  = float(np.dot(E_now, self.alpha_t))   # e_tilde(k), Eq.(15)

        EET = np.outer(self.E_prev, self.E_prev)       # (n-1)x(n-1) matrix
        self.alpha_t = (self.alpha_t
                        - EET @ self.alpha_t
                        + self.E_prev * self.e_prev)   # Eq.(16)

        # Project onto convex simplex
        self.alpha_t = np.clip(self.alpha_t, 0.0, 1.0)
        if self.alpha_t.sum() > 1.0:
            self.alpha_t /= self.alpha_t.sum()

        self.E_prev  = E_now
        self.e_prev  = e_now
        self.states  = [(h.detach(), c.detach()) for h, c in new_states]
        return err


def run_convex_lstm(y, u):
    """
    Online series-parallel identification using the convex cluster.
    """
    cluster = ConvexLSTMCluster()
    err     = np.zeros(T)
    for t in range(2, T):
        x_np   = np.array([y[t-1], y[t-2], u[t-1], u[t-2]], dtype=np.float32)
        err[t] = cluster.step(x_np, y[t])
    return err


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------
def moving_avg(x, w):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - w + 1): i + 1].mean()
    return out


# -----------------------------------------------------------------------
# Git push helper
# Mirrors the logic of the working MATLAB git push block exactly:
#   Step 1  : configure git identity
#   Step 2  : set SSH remote
#   Step 3  : git add .
#   Step 4  : git status
#   Step 5  : git diff --cached --quiet  (check for staged changes)
#   Step 6  : git commit -m "<timestamped message>"
#   Step 7  : git push -u origin <branch>
# -----------------------------------------------------------------------

REPO_SSH = "git@github.com:hzolfaghari2022/LSTM_Modelling.git"


def run_git(cmd_str, check=True):
    """
    Run a shell git command exactly like MATLAB system() does.
    cmd_str  : the full command as a single string, e.g. 'git add .'
    check    : if True, raise SystemExit when the command fails.
    Returns  : (returncode, combined_output_string)
    """
    print(f"  $ {cmd_str}")
    result = subprocess.run(
        cmd_str,
        shell=True,          # use shell=True so 'git add .' works identically to MATLAB system()
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output)
    if check and result.returncode != 0:
        print(f"\n  ERROR: '{cmd_str}' exited with code {result.returncode}.")
        sys.exit(1)
    return result.returncode, output


def git_push_ssh():
    """
    Push to GitHub using SSH, following the same steps as the working
    MATLAB code. Uses 'git add .' to stage everything in the folder,
    checks for actual staged changes before committing, and attaches a
    timestamp to the commit message.
    """
    from datetime import datetime

    print("\n" + "=" * 60)
    print("  Committing and pushing files to GitHub using SSH")
    print("  Repository: hzolfaghari2022/LSTM_Modelling")
    print("=" * 60)

    # Step 1: Configure git identity (same as MATLAB Step 10)
    run_git('git config user.name "Hussein Zolfaghari"')
    run_git('git config user.email "h.zolfaghari2015@gmail.com"')

    # Step 2: Make sure the remote is set to SSH (same URL as MATLAB)
    rc, current_remote = run_git("git remote get-url origin", check=False)
    if rc != 0:
        # No remote exists yet, add it
        run_git(f"git remote add origin {REPO_SSH}")
    elif current_remote.strip() != REPO_SSH:
        # Remote exists but points somewhere else, update it
        run_git(f"git remote set-url origin {REPO_SSH}")
    else:
        print(f"  Remote origin already set to SSH: {REPO_SSH}")

    # Step 3: git add .  (same as MATLAB Step 11 addStatus)
    rc_add, add_out = run_git("git add .", check=False)
    if rc_add != 0:
        print(f"\n  Git add failed:\n{add_out}")
        sys.exit(1)

    # Step 4: git status  (same as MATLAB Step 11 system('git status'))
    run_git("git status", check=False)

    # Step 5: Check for staged changes using git diff --cached --quiet
    #         (same as MATLAB diffStatus check)
    rc_diff, _ = run_git("git diff --cached --quiet", check=False)
    if rc_diff == 0:
        # Nothing staged, nothing to commit
        print("\n  No new changes to commit. Repository is already up to date.")
        return

    # Step 6: Commit with a timestamped message
    #         (same as MATLAB commitMessage = ['Update ... ' datestr(now,...)])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Update Wang 2017 ACC LSTM simulation results - {timestamp}"
    rc_commit, commit_out = run_git(
        f'git commit -m "{commit_message}"', check=False
    )
    if rc_commit != 0:
        print(f"\n  Git commit failed:\n{commit_out}")
        sys.exit(1)
    print(f"\n  Commit completed successfully:\n  {commit_message}")

    # Detect the current branch name automatically
    rc_br, branch = run_git(
        "git rev-parse --abbrev-ref HEAD", check=False
    )
    branch = branch.strip() if rc_br == 0 and branch.strip() else "main"

    # Step 7: git push -u origin <branch>
    #         (same as MATLAB pushCommand = ['git push -u origin ' branchName])
    rc_push, push_out = run_git(
        f"git push -u origin {branch}", check=False
    )
    if rc_push != 0:
        print(
            f"\n  Git push failed using SSH.\n\n"
            f"  If the message says 'Permission denied (publickey)', your SSH key\n"
            f"  is not connected to GitHub yet.\n\n"
            f"  Run this in Git Bash to test:\n"
            f"    ssh -T git@github.com\n\n"
            f"  Git message:\n{push_out}"
        )
        sys.exit(1)

    print(f"\n  Files pushed successfully to GitHub '{branch}' branch.")
    print("=" * 60)
    print("  GitHub SSH push completed for LSTM_Modelling repository.")
    print("=" * 60)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Wang 2017 ACC Simulation 1 with optional GitHub SSH push"
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Commit and push results to GitHub via SSH after the simulation"
    )
    args = parser.parse_args()

    # Resolve the folder where THIS script lives and save all output
    # files there, regardless of which directory PowerShell is currently in.
    # This is the root cause of the push failure: the figures were being
    # saved to the PowerShell working directory (Jupyter_Notebooks), not
    # to the script folder, so git add . found nothing to commit.
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"  Working directory set to: {script_dir}")

    print("=" * 60)
    print("  Wang 2017 ACC   Simulation 1 / Figure 5")
    print("=" * 60)

    print("\n[1/3] Simulating plant (Eq. 18-19) ...")
    y, u = simulate_plant()
    print(f"      y in [{y.min():.3f}, {y.max():.3f}]")

    print("\n[2/3] Online RNN identification (Section III-A) ...")
    rnn_err = run_rnn(y, u)
    print(f"      Mean |e| last 200 steps : {rnn_err[-200:].mean():.5f}")

    print("\n[3/3] Online Convex-LSTM identification (Sections IV, VI) ...")
    lstm_err = run_convex_lstm(y, u)
    print(f"      Mean |e| last 200 steps : {lstm_err[-200:].mean():.5f}")

    print("\n" + "=" * 60)
    print(f"  {'Method':<20} {'Mean|e| all':>13} {'Mean|e| last200':>16}")
    print("  " + "-" * 50)
    for name, e in [("RNN", rnn_err), ("Convex-LSTM", lstm_err)]:
        print(f"  {name:<20} {e[2:].mean():>13.5f} {e[-200:].mean():>16.5f}")

    # Save figures into the script folder (same folder git will commit from)
    k_ax   = np.arange(T)
    rnn_s  = moving_avg(rnn_err,  SMOOTH)
    lstm_s = moving_avg(lstm_err, SMOOTH)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_ax, rnn_s,  color="red",  linewidth=1.0,
            linestyle="--", label="RNN")
    ax.plot(k_ax, lstm_s, color="blue", linewidth=1.0,
            label="Convex-based LSTM")
    ax.set_xlim(0, T)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Identification Errors", fontsize=12)
    ax.set_title(
        "Fig. 5 Reproduction  --  Identification Error for Simulation 1\n"
        "Wang, A New Concept using LSTM for Dynamic System ID, ACC 2017",
        fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    plt.tight_layout()
    plt.show()
    fig1_path = script_dir / "wang2017_fig5_reproduction.png"
    fig.savefig(str(fig1_path), dpi=150, bbox_inches="tight")
    print(f"\n  [Saved] {fig1_path}")

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(k_ax, rnn_err,  color="red",  linewidth=0.5, alpha=0.75)
    ax1.set_ylabel("|error|", fontsize=10)
    ax1.set_title("RNN  (raw, no smoothing)", fontsize=10)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax2.plot(k_ax, lstm_err, color="blue", linewidth=0.5, alpha=0.75)
    ax2.set_ylabel("|error|", fontsize=10)
    ax2.set_xlabel("Time Steps", fontsize=10)
    ax2.set_title("Convex-LSTM  (raw, no smoothing)", fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle=":", alpha=0.5)
    fig2.suptitle("Raw Identification Errors  --  Wang 2017 ACC Simulation 1",
                  fontsize=11)
    plt.tight_layout()
    plt.show()
    fig2_path = script_dir / "wang2017_fig5_raw_errors.png"
    fig2.savefig(str(fig2_path), dpi=150, bbox_inches="tight")
    print(f"  [Saved] {fig2_path}")
    print("\nDone.")

    # Git push (same logic as the working MATLAB doGitHubPush block).
    # Because we already called os.chdir(script_dir) above, all git
    # commands run inside the correct repository folder automatically.
    if args.push:
        git_push_ssh()
    else:
        print("\nGitHub push is disabled. Run with --push to push results.")


if __name__ == "__main__":
    main()