

# ===================================================================
def print_tree(dir_path, indent=""):
    for item in sorted(os.listdir(dir_path)):
        path = os.path.join(dir_path, item)
        if os.path.isdir(path):
            print(f"{indent}📁 {item}")
            print_tree(path, indent + "    ")

# ===================================================================

