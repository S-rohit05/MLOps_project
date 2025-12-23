import os

# usage: running this will replace the artifact locations in meta.yaml files
# with the valid current absolute path to mlruns_xgboost

ROOT_DIR = "mlruns_xgboost"

# Get current absolute path formatted as file URI
# e.g. file:///C:/Users/.../mlruns_xgboost or file:///home/runner/.../mlruns_xgboost
current_path = os.path.abspath(ROOT_DIR).replace("\\", "/")
if not current_path.startswith("/"):
    current_path = "/" + current_path
NEW_URI = f"file://{current_path}"

print(f"Updating MLflow metadata to: {NEW_URI}")

count = 0
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file == "meta.yaml":
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            for line in lines:
                if "artifact_location: " in line:
                    # Replace the entire value after key
                    new_lines.append(f"artifact_location: {NEW_URI}\n")
                    modified = True
                else:
                    new_lines.append(line)
            
            if modified:
                with open(file_path, "w") as f:
                    f.writelines(new_lines)
                count += 1

print(f"Fixed {count} meta.yaml files.")
