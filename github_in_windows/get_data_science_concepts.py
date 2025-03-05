import os

def generate_markdown(folder_name, root_dir):
    # Get all directories in the specified folder
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_name} does not exist in {root_dir}.")
        return []
    
    # List all subdirectories
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    return subdirs

def main():
    # Set the root directory (where your project is located) and output markdown file
    root_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory of the script
    output_file = os.path.join(root_dir, "ds_concepts_list.md")
    
    # List of folder names to generate markdown for
    folders = ["Metrics", "ML_Concepts"]
    
    all_subdirs = []
    
    # Collect all subdirectories from the specified folders
    for folder in folders:
        subdirs = generate_markdown(folder, root_dir)
        # Append each folder's subdirectory along with the folder name (to help generate links)
        all_subdirs.extend([(folder, subdir) for subdir in subdirs])
    
    # Sort the combined list of directories alphabetically
    all_subdirs.sort(key=lambda x: x[1].lower())  # Sort by the subdirectory name in case-insensitive manner
    
    # Write the header to the markdown file
    with open(output_file, 'w') as md_file:
        md_file.write("# Data Science Concepts\n\n")
        
        # Write the links to the markdown file
        for folder, subdir in all_subdirs:
            subdir_link = f"https://github.com/bhishanpdl/Data_Science/tree/master/{folder}/{subdir}"
            md_file.write(f"- [{subdir}]({subdir_link})\n")
        md_file.write("\n")
    
    print(f"Markdown file '{output_file}' generated successfully!")

if __name__ == "__main__":
    main()
