import os
import shutil
import random

def moving_files_with_specific_name(cur_dir = '/Users/SirJerrick/Downloads/data/dogs-vs-cats/dog', dest = '/Users/SirJerrick/Downloads/data/dogs-vs-cats/cat'):
    for root, dir, filename in os.walk(cur_dir):
        for file in filename:
            if file.startswith('cat'):
                shutil.move(os.path.join(root, file), dest)

def randomly_pull_out_percentage(data_dir, out_dir, percent):
    percent = float(percent)
    print("\nmoving {}% of the data from {}, to {}".format(percent, data_dir, out_dir))

    class_dirs = os.listdir(data_dir)

    total_moved = 0
    total_orig = 0

    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        out_class_path = os.path.join(out_dir, class_name)

        os.makedirs(out_class_path)

        class_items = os.listdir(class_path)
        class_items_paths = []

        for class_item in class_items:
            class_item_path = os.path.join(class_path, class_item)
            class_items_paths.append(class_item_path)

        num_orig_items = len(class_items)
        total_orig += num_orig_items

        if num_orig_items:
            num_out = int(float(num_orig_items) / 100.0 * percent)
            if percent == 100:
                num_out = num_orig_items

            random.shuffle(class_items_paths)

            out_item_paths = class_items_paths[:num_out]

            for out_item_path in out_item_paths:
                basename = os.path.basename(out_item_path)

                dest_path = os.path.join(out_class_path, basename)

                os.rename(out_item_path, dest_path)

            total_moved += num_out

    print("\n\tMoved a total of {} out of {} items from \n\t{} \n\tto \n\t{}".format(total_moved, out_dir, total_orig, data_dir))

def make_validation_set_from_train_set(train_path, percent, val_dirname = 'val'):
    parent = os.path.dirname(train_path)
    val_dir_path = os.path.join(parent, val_dirname)

    print("\ncreating val set at \t{}".format(val_dir_path))
    randomly_pull_out_percentage(train_path, val_dir_path, percent)
    return val_dir_path

def main():
    #moving_files_with_specific_name()
    #print("Moved files!")
    make_validation_set_from_train_set('/Users/SirJerrick/Downloads/data/dogs-vs-cats/trainset', percent = 20)

if __name__ == '__main__':
    main()