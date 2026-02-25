"""
Combined argument parser for grpo_textgame.py.
Covers all flags from both grpo_text_action.py and grpo_text_compound.py.
"""

import argparse
import torch


def parse_args():
    """Parse command line arguments for grpo_textgame.py."""
    parser = argparse.ArgumentParser(description="GRPO TextGame - unified text/compound action modes")

    # ── Mode selection ──
    parser.add_argument("--action_mode", type=str, default="text", choices=["text", "compound"],
                        help="Action mode: 'text' (plain words) or 'compound' (JSON helpers)")

    # ── Model settings ──
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--thinking_tokens", type=int, default=256,
                        help="Number of tokens for thinking/reasoning stage")
    parser.add_argument("--action_tokens", type=int, default=128,
                        help="Max new tokens for action/JSON generation stage")

    # ── Text-mode specific ──
    parser.add_argument("--use_two_stage", action="store_true", default=True,
                        help="Use two-stage generation: thinking → action word (text mode only)")
    parser.add_argument("--no_two_stage", action="store_false", dest="use_two_stage",
                        help="Disable two-stage generation")
    parser.add_argument("--logprob_mode", type=str, default="action+thinking",
                        choices=["action", "action+thinking"],
                        help="Log-probability mode (text mode only)")

    # ── Loss settings ──
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "drgrpo"],
                        help="Loss type: 'grpo' (with std normalization) or 'drgrpo' (no std norm)")

    # ── Training settings ──
    parser.add_argument("--num_episodes", type=int, default=800)
    parser.add_argument("--episodes_per_update", type=int, default=8,
                        help="Total episodes per update (single-GPU)")
    parser.add_argument("--episodes_per_gpu", type=int, default=4,
                        help="Episodes per GPU when using multi-GPU")
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--max_env_steps", type=int, default=20)
    parser.add_argument("--eat_reward", type=float, default=1.0,
                        help="Reward for eating an apple (default: 1.0)")
    parser.add_argument("--clean_reward", type=float, default=0.0,
                        help="Reward for cleaning a dirt tile (default: 0.0)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # ── Inner epoch optimization ──
    parser.add_argument("--num_inner_epochs", type=int, default=4,
                        help="Number of optimization epochs per group (default: 4)")
    parser.add_argument("--minibatch_size", type=int, default=8,
                        help="Number of trajectories per mini-batch (default: 8)")
    parser.add_argument("--samples_per_micro_batch", type=int, default=2,
                        help="Samples per micro-batch for gradient accumulation (default: 2)")
    parser.add_argument("--micro_batch_size", type=int, default=8,
                        help="DEPRECATED - Use --samples_per_micro_batch instead")

    # ── Checkpoint settings ──
    parser.add_argument("--output_dir", type=str, default="./grpo_textgame_checkpoints")
    parser.add_argument("--save_steps", type=int, default=50)

    # ── Device / quantization ──
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--use_accelerate", action="store_true", default=False,
                        help="Use Accelerate for multi-GPU training")
    parser.add_argument("--use_deepspeed", action="store_true", default=False,
                        help="Enable DeepSpeed ZeRO-3 sharding (requires --use_accelerate)")

    # ── Evaluation settings ──
    parser.add_argument("--eval_interval", type=int, default=128,
                        help="Evaluate every N training episodes (0 = disabled)")
    parser.add_argument("--num_eval_episodes", type=int, default=20)
    parser.add_argument("--skip_pre_eval", action="store_true", default=False)
    parser.add_argument("--skip_post_eval", action="store_true", default=False)

    # ── Episode trajectory logging ──
    parser.add_argument("--log_trajectory", action="store_true", default=True,
                        help="Log one random episode per update to file")
    parser.add_argument("--no_log_trajectory", action="store_false", dest="log_trajectory")
    parser.add_argument("--trajectory_log_file", type=str, default="episode_trajectories.txt")

    # ── Wandb settings ──
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb_project", type=str, default="grpo_textgame")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # ── Visualization mode ──
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Visualize a rollout (skip training)")
    parser.add_argument("--viz_save_file", type=str, default=None,
                        help="Save visualization to file")
    parser.add_argument("--viz_use_ref", action="store_true", default=False,
                        help="Use reference model for visualization")

    # ── Sanity check mode ──
    parser.add_argument("--sanity_check", action="store_true", default=False,
                        help="Run sanity check: 1 agent, 50 episodes")

    return parser.parse_args()
