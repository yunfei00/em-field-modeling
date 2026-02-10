import os
import sys

def main():
    # usage: python scripts/find_latest_run.py runs demo
    base = sys.argv[1] if len(sys.argv) > 1 else "runs"
    exp = sys.argv[2] if len(sys.argv) > 2 else "demo"
    exp_dir = os.path.join(base, exp)
    if not os.path.isdir(exp_dir):
        print("")
        return

    run_ids = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if not run_ids:
        print("")
        return

    run_ids.sort()
    latest = run_ids[-1]
    print(os.path.join(exp_dir, latest))

if __name__ == "__main__":
    main()