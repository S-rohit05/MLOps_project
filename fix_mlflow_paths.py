import os

# usage: running this will replace the absolute Windows paths in meta.yaml files
# with the internal docker path '/mlruns'

ROOT_DIR = "mlruns_xgboost"
SEARCH_STRING = "file:C:/Users/sanap/.gemini/antigravity/scratch/mlops_project/mlruns_xgboost"
REPLACE_STRING = "file:///mlruns"

count = 0
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file == "meta.yaml":
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()
            
            if SEARCH_STRING in content:
                new_content = content.replace(SEARCH_STRING, REPLACE_STRING)
                with open(file_path, "w") as f:
                    f.write(new_content)
                print(f"Updated {file_path}")
                count += 1
            else:
                # Also try matching standard windows path without file:/// prefix just in case
                # or different casing.
                # Actually, let's just use a more generic replace for current dir
                pass

print(f"Fixed {count} meta.yaml files.")
