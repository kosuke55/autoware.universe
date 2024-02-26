import os
import re


def replace_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Replace 'perception' and 'perception' with 'perception' and 'Perception', keeping the case
    new_content = re.sub(
        r"perception([a-zA-Z_]*)", lambda m: "perception" + m.group(1), content, flags=re.IGNORECASE
    )
    new_content = re.sub(
        r"perception([a-zA-Z_]*)", lambda m: "Perception" + m.group(1), new_content
    )

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(new_content)


def process_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        # Replace in files
        for name in files:
            file_path = os.path.join(root, name)
            replace_content(file_path)

        # Rename files and directories if needed
        all_items = dirs + files
        for name in all_items:
            original_path = os.path.join(root, name)
            new_name = re.sub(
                r"perception([a-zA-Z_]*)",
                lambda m: "perception" + m.group(1),
                name,
                flags=re.IGNORECASE,
            )
            new_name = re.sub(
                r"perception([a-zA-Z_]*)", lambda m: "Perception" + m.group(1), new_name
            )
            new_path = os.path.join(root, new_name)
            if original_path != new_path:
                os.rename(original_path, new_path)
                print(f"Renamed {original_path} to {new_path}")


# Correct the base directory path according to your system's actual path
base_directory = (
    "/home/kosuke55/pilot-auto.latest/src/autoware/universe/evaluator/perception_evaluator"
)

process_directory(base_directory)
