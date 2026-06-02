"""Quick test: does build_robofac_records() find videos after re-download?"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

base = DATA_DIR / "RoboFAC"
json_path = base / "training_qa.json"

with open(json_path) as f:
    data = json.load(f)

print(f"Total JSON entries: {len(data)}")

found_rw, found_sim, missing = 0, 0, 0
missing_examples = []

for raw in data:
    v = raw.get("video", "")
    if not v:
        continue    
    rw_path  = base / "realworld_data" / v
    sim_path = base / "simulation_data" / v
    if rw_path.exists():
        found_rw += 1
    elif sim_path.exists():
        found_sim += 1
    else:
        missing += 1
        if len(missing_examples) < 3:
            missing_examples.append(v)

print(f"realworld_data matches:  {found_rw}")
print(f"simulation_data matches: {found_sim}")
print(f"missing:                 {missing}")
print(f"TOTAL found:             {found_rw + found_sim}")

if missing_examples:
    print(f"\nSample missing paths:")
    for p in missing_examples:
        print(f"  {p}")

if found_rw + found_sim > 0:
    print("\n✅ build_robofac_records() should work!")
else:
    print("\n❌ No videos found — build_robofac_records() will return 0 records")
