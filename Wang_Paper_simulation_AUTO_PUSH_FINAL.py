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

GitHub push:
    The script automatically commits and pushes the updated results to
    git@github.com:hzolfaghari2022/LSTM_Modelling.git after each run.

To run without pushing:
    python wang2017_simulation.py --no-push
 
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
# GitHub automatic push helper
# -----------------------------------------------------------------------
# This block automatically commits and pushes all updated files in the same
# folder as this Python script to the GitHub repository below.
# It uses SSH, so your GitHub SSH key must already work:
#     ssh -T git@github.com
# -----------------------------------------------------------------------

REPO_SSH = "git@github.com:hzolfaghari2022/LSTM_Modelling.git"
BRANCH_NAME = "main"


def run_git(cmd_str, check=True):
    """
    Run a Git command and print its output.
    """
    print(f"  $ {cmd_str}")
    result = subprocess.run(
        cmd_str,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    output = (result.stdout + result.stderr).strip()
    if output:
        print(output)

    if check and result.returncode != 0:
        print(f"\n  ERROR: Git command failed:\n  {cmd_str}")
        sys.exit(1)

    return result.returncode, output


def git_push_to_github():
    """
    Automatically push the current script folder to GitHub after each run.

    Steps:
      1. Work from the folder where this script is saved.
      2. Make sure Git is initialized.
      3. Set the SSH remote to the LSTM_Modelling repository.
      4. Stage all new/updated files.
      5. Commit if there are changes.
      6. Pull remote main safely.
      7. Push to origin/main.
    """
    from datetime import datetime

    print("\n" + "=" * 60)
    print("  Automatic GitHub push")
    print("  Repository: hzolfaghari2022/LSTM_Modelling")
    print("=" * 60)

    # Always run Git from the folder where this Python file is located.
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"  Git working folder: {script_dir}")

    # Check Git installation.
    run_git("git --version")

    # Initialize this folder as a Git repository if needed.
    rc_repo, _ = run_git("git rev-parse --is-inside-work-tree", check=False)
    if rc_repo != 0:
        print("  This folder is not a Git repository yet. Initializing it now...")
        run_git("git init")

    # If a merge is unfinished, stop and tell the user exactly what to do.
    git_dir_rc, git_dir = run_git("git rev-parse --git-dir", check=False)
    if git_dir_rc == 0:
        merge_head = Path(git_dir.strip()) / "MERGE_HEAD"
        if merge_head.exists():
            print("\n  Git found an unfinished merge in this folder.")
            print("  Finish it manually once, then run the Python code again:")
            print('    git status')
            print('    git commit -m "Merge remote main into local main"')
            print('    git push -u origin main')
            sys.exit(1)

    # Configure Git identity.
    run_git('git config user.name "Hussein Zolfaghari"')
    run_git('git config user.email "h.zolfaghari2015@gmail.com"')

    # Make sure origin points to the correct SSH repository.
    rc_remote, current_remote = run_git("git remote get-url origin", check=False)
    if rc_remote != 0:
        run_git(f"git remote add origin {REPO_SSH}")
    elif current_remote.strip() != REPO_SSH:
        run_git(f"git remote set-url origin {REPO_SSH}")
    else:
        print(f"  Remote origin is already correct: {REPO_SSH}")

    # Always use the main branch locally.
    run_git(f"git branch -M {BRANCH_NAME}", check=False)

    # Stage all files in this folder.
    rc_add, add_out = run_git("git add .", check=False)
    if rc_add != 0:
        print(f"\n  Git add failed:\n{add_out}")
        sys.exit(1)

    # Show status for transparency.
    run_git("git status", check=False)

    # Commit only if something actually changed.
    rc_diff, _ = run_git("git diff --cached --quiet", check=False)
    if rc_diff == 0:
        print("\n  No new local changes to commit.")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Update LSTM simulation results - {timestamp}"
        rc_commit, commit_out = run_git(
            f'git commit -m "{commit_message}"',
            check=False,
        )
        if rc_commit != 0:
            print(f"\n  Git commit failed:\n{commit_out}")
            sys.exit(1)
        print(f"\n  Commit completed successfully:\n  {commit_message}")

    # Pull remote main before pushing. --no-edit prevents Git from opening
    # a text editor for the merge message.
    rc_pull, pull_out = run_git(
        f"git pull origin {BRANCH_NAME} --allow-unrelated-histories --no-rebase --no-edit",
        check=False,
    )
    if rc_pull != 0:
        # If the remote branch does not exist yet, it is okay to continue.
        if "couldn't find remote ref" in pull_out.lower() or "could not find remote ref" in pull_out.lower():
            print("  Remote main branch does not exist yet. Continuing to push a new main branch.")
        else:
            print("\n  Git pull failed. There may be a conflict.")
            print("  Run this manually in PowerShell from the same folder:")
            print("    git status")
            print("  If Git says conflicts exist, fix them first.")
            print("  If Git says all conflicts are fixed, run:")
            print('    git commit -m "Merge remote main into local main"')
            print("    git push -u origin main")
            print("\n  Git message:")
            print(pull_out)
            sys.exit(1)

    # Push to GitHub main.
    rc_push, push_out = run_git(f"git push -u origin {BRANCH_NAME}", check=False)
    if rc_push != 0:
        print("\n  Git push failed.")
        print("  Most common reasons:")
        print("    1. SSH key is not connected to GitHub.")
        print("    2. Remote main has changes that need to be pulled first.")
        print("\n  Test SSH in Git Bash:")
        print("    ssh -T git@github.com")
        print("\n  Git message:")
        print(push_out)
        sys.exit(1)

    print("\n  Files pushed successfully to GitHub main branch.")
    print("=" * 60)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Wang 2017 ACC Simulation 1 with optional GitHub SSH push"
    )
    # By default, this script pushes the updated results to GitHub after each run.
    # Use --no-push only when you want to run the simulation without uploading results.
    parser.add_argument(
        "--no-push", action="store_true",
        help="Run the simulation without committing/pushing results to GitHub"
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
 
    # Git push is enabled by default. Because we already called
    # os.chdir(script_dir) above, all git commands run inside the
    # same folder where the script and output figures are saved.
    if args.no_push:
        print("\nGitHub push is disabled because you used --no-push.")
    else:
        git_push_to_github()
 
 
if __name__ == "__main__":
    main()