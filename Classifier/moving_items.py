import shutil
import os
import fnmatch
from PIL import Image

def moving_folders_with_specific_name(cur_dir, dest):
    """ Parameters
        
        cur_dir = path where your folders want moved
        
        dest = path to destination folder
    
    """
    count = 0
    for root, dirs, filename in os.walk(cur_dir):
        for dir in dirs:
            if dir.startswith('name_you_want'):
                dest_folder = os.path.join(dest, dir + '_' + str(count))
                count += 1
                shutil.copytree(os.path.join(root, dir), dest_folder, symlinks=True)

def moving_files_with_specific_end(data_dir, dest):
                                   
     """ Parameters
        
        cur_dir = path where your folders want moved
        
        dest = path to destination folder
        
     """

    for root, dirs, filename in os.walk(data_dir):
        for files in filename:
            if fnmatch.fnmatch(files, '*.png') and not fnmatch.fnmatch(files, '*Masked.png') and not fnmatch.fnmatch(files, '*MaskedFinal.png') and not fnmatch.fnmatch(files, '*MaskedReplacedBG.png'):
                dest_folder = os.path.join(dest, files)
                shutil.copy(os.path.join(root, files), dest_folder)

def main():
    moving_files_with_specific_end()
    #moving_folders_with_specific_name()

if __name__ == '__main__':
    main()
