from pathlib import Path
from prepare_data import prepare_datasets

# Use a small test catalog for fast verification
test_catalog_path = Path("/tmp/hf_test_catalog.yaml")

# Just pick the first dataset from the generated catalog
with open("scripts/e2e_realdata/hf_dataset_catalog.yaml") as f:
    import yaml
    full_cat = yaml.safe_load(f)

test_cat = {"datasets": [full_cat["datasets"][0]]}

with open(test_catalog_path, "w") as f:
    yaml.dump(test_cat, f)

print(f"Testing with catalog: {test_cat['datasets'][0]['name']}")

try:
    result = prepare_datasets(
        mode="pr", # mode='pr' sample_size=1
        catalog_path=test_catalog_path,
        cache_dir=Path("/tmp/e2e_cache"),
        include_kaggle=False,
        require_kaggle=False,
        seed=42,
        context_cap=512,
        timeout_seconds=30
    )
    
    print("\nSuccess! Prepared datasets:")
    for ds_name, series_list in result.datasets.items():
        print(f"  {ds_name}: {len(series_list)} series")
        for s in series_list:
            print(f"    ID: {s['id']}")
            print(f"    History: {len(s['target'])} points")
            print(f"    Horizon: {len(s['actuals'])} points")
            
except Exception as e:
    print(f"\nError: {e}")
