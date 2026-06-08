import os
import sys
import subprocess

def main():
    print("============================================================")
    print(" Starting ThinkFlow-VLA Stage 2 Mini Training")
    print(" Dataset Subset  : 15%")
    print(" Total Steps     : 100")
    print(" Warmup Steps    : 50")
    print("============================================================")

    # Replace with the actual path to your Stage 1 checkpoint
    stage1_ckpt = "shreethar/stage1_unsloth"
    output_dir = "checkpoints/stage2_mini"

    # Create the output directory for this test run
    os.makedirs(output_dir, exist_ok=True)

    # Base directory for the stage2 module
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "training", "train_stage2.py")

    cmd = [
        sys.executable, train_script,
        "--stage1_ckpt", stage1_ckpt,
        "--output_dir", output_dir,
        "--split", "test",
        "--subset_ratio", "0.15",
        "--total_steps", "100",
        "--warmup_steps", "50",
        "--save_steps", "50",
        "--wandb_run", "stage2-mini-test",
        "--batch_size", "4",
        "--num_workers", "2"
    ]

    print(f"Running command:\n{' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\nMini training complete! Checkpoints saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
