import os

def explore_project_structure(root_dir):
    """
    Comprehensive project structure exploration
    
    Args:
        root_dir (str): Root directory path of the project
    
    Returns:
        dict: Detailed project structure
    """
    project_structure = {
        'root': root_dir,
        'directories': {},
        'files': [],
        'summary': {
            'total_directories': 0,
            'total_files': 0,
            'file_types': {}
        }
    }

    # Walk through directory
    for root, dirs, files in os.walk(root_dir):
        # Print current directory for debugging
        print(f"Checking directory: {root}")  # Debugging line
        
        # Skip version control and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        
        # Relative path from project root
        relative_path = os.path.relpath(root, root_dir)
        
        # Current directory details
        current_dir_info = {
            'path': relative_path,
            'subdirectories': dirs,
            'files': []
        }

        # Process files
        for file in files:
            # Skip certain file types or temporary files
            if not file.startswith(('.', '__')) and not file.endswith(('.pyc', '~')):
                file_path = os.path.join(root, file)
                file_info = {
                    'name': file,
                    'size': os.path.getsize(file_path),
                    'extension': os.path.splitext(file)[1]
                }
                
                current_dir_info['files'].append(file_info)

                # Update file type summary
                ext = file_info['extension']
                project_structure['summary']['file_types'][ext] = project_structure['summary']['file_types'].get(ext, 0) + 1

        # Store directory info
        if relative_path != '.':
            project_structure['directories'][relative_path] = current_dir_info
        
        # Update summary
        project_structure['summary']['total_directories'] += len(dirs)
        project_structure['summary']['total_files'] += len(files)

    return project_structure

def print_project_structure(structure):
    """
    Pretty print project structure
    
    Args:
        structure (dict): Project structure dictionary
    """
    print("Project Structure Overview:")
    print(f"Root Directory: {structure['root']}")
    print("\nSummary:")
    print(f"Total Directories: {structure['summary']['total_directories']}")
    print(f"Total Files: {structure['summary']['total_files']}")
    
    print("\nFile Types:")
    for ext, count in structure['summary']['file_types'].items():
        print(f"  {ext}: {count}")
    
    print("\nDirectory Details:")
    for path, dir_info in structure['directories'].items():
        print(f"\n{path}:")
        print(f"  Subdirectories: {dir_info['subdirectories']}")
        print("  Files:")
        for file in dir_info['files']:
            print(f"    - {file['name']} (Size: {file['size']} bytes)")

# Usage
if __name__ == "__main__":
    project_dir = r'D:\\Work\\PortfolioProjects\\1\\BraineeClassifier'  # Update with your actual project path
    structure = explore_project_structure(project_dir)
    print_project_structure(structure)
