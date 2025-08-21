# -*- coding: utf-8 -*-

import zipfile
import os

def zip_dir(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file with a relative path to avoid full system path in zip
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

# Example usage
zip_dir('path/to/folder', 'output.zip')