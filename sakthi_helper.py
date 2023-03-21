import os
import pathlib
import re
import shutil

import cv2
import numpy as np
import pandas as pd
from keras.applications import VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, Xception, MobileNet, DenseNet121, \
    NASNetMobile, EfficientNetB0
from keras.layers import BatchNormalization
from keras_preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dropout, Dense, MaxPooling2D
from tensorflow.python.keras.saving.model_config import model_from_yaml


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def preprocess_image_for_CNN(input_image, target_size, target_type='rgb'):

    resized_image = cv2.resize(input_image, target_size)

    if target_type == "rgb":
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = resized_image

    img = img_to_array(rgb_image)

    img /= 255

    return img


def print_file_details(dir_path, file_name):
    print("----------------------")
    print('file name : ', file_name)
    file_path = os.path.join(dir_path, file_name)
    print("file path : ", file_path)
    print("file name : ", os.path.basename(file_path))
    path_before_ext, file_extension = os.path.splitext(file_path)
    folder = path_before_ext.split(os.sep)[-2]
    print("folder : ", folder)
    just_file_name = path_before_ext.split(os.sep)[-1]
    print("just_file_name : ", just_file_name)
    print("file extension : ", file_extension)
    
def print_oswalk_details(dir_path, sub_dirs, file_names):
    print("=====================")
    print('The current directory path is ' + dir_path)
    dir_name = os.path.basename(dir_path)
    print("dir name : ", dir_name)
    parent_dir_path = os.path.dirname(dir_path)
    print("parent dir path : ", parent_dir_path)
    num_subfolders = len(sub_dirs)
    print("num subfolders : ", num_subfolders)
    print("SUBFOLDERS :", sub_dirs)
    num_files = len(file_names)
    print("num files:  ", num_files)
    print("FILES : ", file_names)
    print("=====================")


def flatten_2D_list(list):
    """
    Flatten a 2D list into a 1D list.

    Args:
    lst: The 2D list to flatten.

    Returns:
    A flattened 1D list.
    """
    return [elem for sublist in list for elem in sublist]


def get_grouped_ids_from_df(df, grouping_cols=["speed", "angle"], id_col="image_id", inter_grouping=False):

    grouped_image_ids = {}
    for group in grouping_cols:
        unique_values = sorted(df[group].unique())
        grouped_image_ids[group] = {}
        for value in unique_values:
            image_ids = df.loc[df[group] == value, [id_col]].astype('str').values
            image_ids = flatten_2D_list(image_ids)
            grouped_image_ids[group][value] = image_ids

    if inter_grouping:
        pass

    return grouped_image_ids


def execute_path_split_filter(path_splits, path_split_filters):
    filter_out = False
    for i, filter in enumerate(path_split_filters):
        # print(i, filter_out, filter[1], path_splits[filter[0]])
        if (filter_out or i == 0) and (filter[1] == path_splits[filter[0]]):
            # print("---", filter[1], path_splits[filter[0]])
            filter_out = True
        else:
            filter_out = False

    return filter_out

def get_last_id(dir_path):
    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    print("Total files in the dir : ", len(all_files))
    id_list = []
    for file_name in all_files:
        id = int(file_name.split("_")[0])
        id_list.append(id)


    return max(id_list)



def get_normalized_speed_and_angle(speed, angle):

    norm_speed = (speed-0)/35
    norm_angle = (angle-50)/80

    return norm_speed, norm_angle

def backup_files(src_dir, dest_dir):

    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)  #dirs_exist_ok=True - this will overwrite the files if file already exists - be cautious
    print("All files from the source directory  - {} backed up to the destination directory - {}".format(src_dir, dest_dir))


def get_cleaned_data(dir_path, format_change_str, id_start_from=15000, test_mode=True):

    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    print("Total files in the dir : ", len(all_files))

    id = id_start_from
    old_format, new_format = format_change_str.split("-")
    new_dir_path = os.path.dirname(dir_path) + "\\cleaned_{}".format(os.path.basename(dir_path))
    new_path_list = []
    copy_count = 0
    for file_name in all_files:
        id += 1
        old_file_path = os.path.join(dir_path, file_name)
        print("Old_file_path : ", old_file_path)
        file_name, ext = file_name.split(".")

        time_stamp, angle, speed = file_name.split("_")

        norm_speed, norm_angle = get_normalized_speed_and_angle(int(speed), int(angle))

        new_file_name = "{}_speed-{}_angle-{}.png".format(id, norm_speed, norm_angle)

        pathlib.Path(new_dir_path).mkdir(parents=True, exist_ok=True)

        new_file_path = os.path.join(new_dir_path, new_file_name)
        print("New_file_path : ", new_file_path)
        new_path_list.append(new_file_path)
        if not test_mode:
            shutil.copy2(old_file_path, new_file_path)
            copy_count += 1

    print("Total files copied : ", copy_count)

    return new_path_list, id



def check_data_sanity(csv_path, image_dir):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Get a list of image files in the image directory
    files_and_dirs = os.listdir(image_dir)
    print("Total files and directories under image dir : ", len(files_and_dirs))
    
    dir_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    print("Total files in the image dir : ", len(dir_files))
    
    # Get the set of image IDs from the CSV
    csv_image_ids = set(df["image_id"].astype('str'))
    print("Number of image_ids in CSV : ", len(csv_image_ids))
    
    # Get the set of image IDs from the image directory
    dir_image_ids = set(os.path.splitext(file)[0] for file in dir_files)
    print("Number of images in image dir : ", len(dir_image_ids))
    
    # Find the image IDs that are in the CSV but not in the image directory
    missing_from_image_dir = list(csv_image_ids - dir_image_ids)
    print("the image IDs that are in the CSV but not in the image directory : ", len(missing_from_image_dir))
    
    # Find the image files that are in the image directory but not in the CSV
    missing_from_csv = [file for file in dir_files if os.path.splitext(file)[0] not in csv_image_ids]
    print("the image files that are in the image directory but not in the CSV : ", len(missing_from_csv))
    
    return missing_from_image_dir, missing_from_csv


def traverse_directories(root_dir, ignore_dirs=["^\..*", "corrupted_files"], filter_extensions=[".png", ".jpg"], verbose=True):
    """
    Traverse through directories under root_dir recursively and prints the folder name, subfolder names and filenames.

    root_dir: str, path to the root directory
    ignore_dirs: list of str, regular expressions for directory names to ignore

    ["^\..*"] - regex that matches folder names that start with a period followed by any other characters

    """
    for dir_path, sub_dirs, file_names in os.walk(root_dir):

        if verbose:
            print_oswalk_details(dir_path, sub_dirs, file_names)

        dir_name = os.path.basename(dir_path)

        if any(re.match(pattern, dir_name) for pattern in ignore_dirs):
            print("Skipping directory ", dir_path)
            sub_dirs.clear()
            print("Skipping all sub directories - ", sub_dirs)
            continue

        file_count = 0
        for file_name in file_names:
            file_count += 1

            if verbose:
                if file_count < 10:
                    print_file_details(dir_path, file_name)
                else:
                    continue


def move_to_corrupted(file_path, test_mode, move_flag=False):
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    corrupted_dir = os.path.join(parent_dir, "corrupted files")
    os.makedirs(corrupted_dir,
                exist_ok=True)  ### alternative - pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
    file_name = os.path.basename(file_path)
    new_file_path = os.path.join(corrupted_dir, file_name)
    if test_mode:
        print("new file path - {}".format(new_file_path))
    else:
        shutil.move(file_path, new_file_path)
        print("corrupted file {} moved to {}".format(file_name, new_file_path))
        move_flag = True
    print("------------------")

    return new_file_path, move_flag


def clean_file(file_path, test_mode, move_flag=False):
    new_file_path = False
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print("File {} is empty".format(file_path))
        new_file_path, move_flag = move_to_corrupted(file_path, test_mode)
    else:
        try:
            with open(file_path, 'rb') as f:
                f.read(1)
        except:
            print("File {} is corrupted".format(file_path))
            new_file_path, move_flag = move_to_corrupted(file_path, test_mode)

    return new_file_path, move_flag


def move_corrupted_files(root_dir, data_dirs=["training_data"], ignore_dirs=["^\..*", "corrupted files"], filter_extensions=[".png", ".jpg"],
                             test_mode=True, verbose=False):
    """

    """
    all_corrupted_files = []
    total_file_count = 0
    total_filtered_file_count = 0
    total_corrupted_count = 0
    total_move_count = 0
    for dir_path, sub_dirs, file_names in os.walk(root_dir):

        if verbose:
            print_oswalk_details(dir_path, sub_dirs, file_names)

        dir_name = os.path.basename(dir_path)

        #----------- filtering out directories that match the ignore_dirs parameter --------------
        if any(re.match(pattern, dir_name) for pattern in ignore_dirs):
            print("Checking ignore_dir patterns - {}".format(ignore_dirs))
            print("Skipping directory ", dir_path)
            sub_dirs.clear()
            print("skipping all sub directories - ", sub_dirs)
            continue

        #------------ filtering only the directries in data_dir parameter -------------------------
        for dir in data_dirs:
            if dir_name != dir:
                print("Checking if its a data directory")
                print("Skipping directory ", dir_path)
                continue

        file_count = 0
        for file_name in file_names:
            file_count += 1
            total_file_count += 1
            file_path = os.path.join(dir_path, file_name)
            # print("file path : ", file_path)
            path_before_ext, file_ext = os.path.splitext(file_path)

            #------------- filtering only the files with extensions in filter_extensions parameter -----------
            if file_ext in filter_extensions:
                total_filtered_file_count += 1
                # print("filtered file : ", file_path)
                new_file_path, move_flag = clean_file(file_path, test_mode)
                if move_flag:
                    total_move_count += 1
                if new_file_path:
                    total_corrupted_count += 1
                    file_name = os.path.basename(file_path).split(".")[0]
                    all_corrupted_files.append((new_file_path, (dir_name, file_name)))

            if verbose:
                if file_count < 10:
                    print_file_details(dir_path, file_name)
                else:
                    continue

    print("All corrupted files are moved successfully")

    print("Total files count : ", total_file_count)
    print("Total filtered files count : ", total_filtered_file_count)
    print("Total corrupted files count : ", total_corrupted_count)
    print("Total moved files count : ", total_move_count)

    return all_corrupted_files

def rename_files(root_dir, rename_df, rename_cols=["speed", "angle"], data_dirs=["training_data"],
                 path_split_filters=[(-3,"mlis2_car"), (-2,"new_data")], ignore_dirs=["^\..*", "corrupted files"],
                           filter_extensions=[".png", ".jpg"], test_mode=True, verbose=False):

    total_source_file_count = 0
    filtered_source_file_count = 0
    total_copy_count = 0
    for dir_path, sub_dirs, file_names in os.walk(root_dir):

        if verbose:
            print_oswalk_details(dir_path, sub_dirs, file_names)

        dir_name = os.path.basename(dir_path)

        # ----------- filtering out directories that match the ignore_dirs parameter --------------
        if any(re.match(pattern, dir_name) for pattern in ignore_dirs):
            print("Skipping directory ", dir_path)
            sub_dirs.clear()
            print("skipping all sub directories - ", sub_dirs)
            continue

        # ------------ filtering only the directries in data_dir parameter -------------------------
        for dir in data_dirs:
            if dir_name != dir:
                print("Checking if its a data directory")
                print("Skipping directory ", dir_path)
                # sub_dirs.clear()
                # print("Skipping all sub directories - ", sub_dirs)
                continue

        file_count = 0
        for file_name in file_names:
            file_count += 1
            total_source_file_count += 1
            file_path = os.path.join(dir_path, file_name)
            # print("file path : ", file_path)
            path_before_ext, file_ext = os.path.splitext(file_path)

            if file_ext in filter_extensions:
                # print("filtered file : ", file_path)
                path_splits = file_path.split(os.sep)
                # print(path_splits)
                file_name = path_splits[-1]

                dir_filter_out = execute_path_split_filter(path_splits, path_split_filters)

                if dir_filter_out:
                    id, ext = file_name.split(".")

                    if int(id) in rename_df["image_id"].values:
                        filtered_source_file_count += 1

                        src_path = file_path
                        dest_dir = root_dir + copy_dir
                        # print(dest_dir)
                        pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

                        image_id_match_condition = rename_df["image_id"] == int(id)

                        rename_params = []
                        rename_params = rename_df.loc[image_id_match_condition, rename_cols].values[0]
                        new_file_name = id
                        for i, param in enumerate(rename_params):
                            new_file_name += "_" + rename_cols[i] + "-" + str(param)
                        dest_path = dest_dir + "\\" + new_file_name + ".{}".format(ext)
                        # print(dest_path)

                        if not (os.path.exists(dest_path)):
                            if file_count < 10:
                                print("SRC : ", src_path)
                                print("DEST : ", dest_path)
                            if not test_mode:
                                shutil.copy2(src_path, dest_path)
                                print("Copied")
                                total_copy_count += 1
                        else:
                            print("{} File already exists. So not copied.")

                if verbose:
                    if file_count < 10:
                        print_file_details(dir_path, file_name)
                    else:
                        continue

    print("All relevant files are copied successfully into respective folders")
    print("total source files count : ", total_source_file_count)
    print("total filtered source files count : ", filtered_source_file_count)
    print("total_copy_count : ", total_copy_count)

def copy_files_dir_to_dir(root_dir, rename_df=None, rename_cols=["speed", "angle"], data_dirs=["training_data"],
                          copy_dir="\\data\\all_data", path_split_filters=[(-2,"training_data"), (-1,"training_data")],
                          ignore_dirs=["^\..*", "corrupted files"],
                           filter_extensions=[".png", ".jpg"], test_mode=True, verbose=False):

    total_source_file_count = 0
    filtered_source_file_count = 0
    total_copy_count = 0
    for dir_path, sub_dirs, file_names in os.walk(root_dir):

        if verbose:
            print_oswalk_details(dir_path, sub_dirs, file_names)

        dir_name = os.path.basename(dir_path)

        # ----------- filtering out directories that match the ignore_dirs parameter --------------
        if any(re.match(pattern, dir_name) for pattern in ignore_dirs):
            print("Skipping directory ", dir_path)
            sub_dirs.clear()
            print("skipping all sub directories - ", sub_dirs)
            continue

        # ------------ filtering only the directries in data_dir parameter -------------------------
        if data_dirs:
            for dir in data_dirs:
                if dir_name != dir:
                    print("Checking if its a data directory")
                    print("Skipping directory ", dir_path)
                    # sub_dirs.clear()
                    # print("Skipping all sub directories - ", sub_dirs)
                    continue

        file_count = 0
        for file_name in file_names:
            file_count += 1
            total_source_file_count += 1
            file_path = os.path.join(dir_path, file_name)
            # print("file path : ", file_path)
            path_before_ext, file_ext = os.path.splitext(file_path)

            if file_ext in filter_extensions:
                # print("filtered file : ", file_path)
                path_splits = file_path.split(os.sep)
                # print(path_splits)
                file_name = path_splits[-1]
                file_name_without_ext = ".".join(file_name.split(".")[:-1])

                # print(path_splits)
                dir_filter_out = execute_path_split_filter(path_splits, path_split_filters)

                if dir_filter_out:
                    # print("FILTERED DIR - {}".format(path_splits))
                    ext = file_name.split(".")[-1]
                    id = file_name_without_ext.split(".")[0]
                    image_labels = "_".join(file_name_without_ext.split("_")[1:])

                    filtered_source_file_count += 1

                    src_path = file_path
                    dest_dir = root_dir + copy_dir
                    # print(dest_dir)
                    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

                    if rename_df:
                        if int(id) in rename_df.get("image_id").values:
                            image_id_match_condition = rename_df["image_id"] == int(id)

                            rename_params = []
                            rename_params = rename_df.loc[image_id_match_condition, rename_cols].values[0]
                            new_file_name = id
                            for i, param in enumerate(rename_params):
                                new_file_name += "_" + rename_cols[i] + "-" + str(param)
                    else:
                        # dest_dir += "{}".format(image_labels)
                        new_file_name = file_name_without_ext

                    dest_path = dest_dir + "\\" + new_file_name + ".{}".format(ext)
                    # print(dest_path)

                    if not (os.path.exists(dest_path)):
                        if file_count < 10:
                            print("SRC : ", src_path)
                            print("DEST : ", dest_path)
                        if not test_mode:
                            shutil.copy2(src_path, dest_path)
                            print("Copied")
                            total_copy_count += 1
                    else:
                        print("{} File already exists. So not copied.")

                if verbose:
                    if file_count < 10:
                        print_file_details(dir_path, file_name)
                    else:
                        continue

    print("All relevant files are copied successfully into respective folders")
    print("total source files count : ", total_source_file_count)
    print("total filtered source files count : ", filtered_source_file_count)
    print("total_copy_count : ", total_copy_count)


def copy_files_one_to_many(root_dir, grouped_ids, rename_df, data_dirs=["training_data"], copy_dir="\\data",
                           path_split_filters=[(-3,"training_data"), (-2,"training_data")],
                           ignore_dirs=["^\..*", "corrupted files"],
                           filter_extensions=[".png", ".jpg"], test_mode=True, verbose=False):

    total_source_file_count = 0
    filtered_source_file_count = 0
    total_copy_count = 0
    for dir_path, sub_dirs, file_names in os.walk(root_dir):

        if verbose:
            print_oswalk_details(dir_path, sub_dirs, file_names)

        dir_name = os.path.basename(dir_path)

        # ----------- filtering out directories that match the ignore_dirs parameter --------------
        if any(re.match(pattern, dir_name) for pattern in ignore_dirs):
            print("Skipping directory ", dir_path)
            sub_dirs.clear()
            print("skipping all sub directories - ", sub_dirs)
            continue

        # ------------ filtering only the directries in data_dir parameter -------------------------
        for dir in data_dirs:
            if dir_name != dir:
                print("Checking if its a data directory")
                print("Skipping directory ", dir_path)
                continue

        for group, values in grouped_ids.items():
            for category, filtered_ids in values.items():
                file_count = 0
                for file_name in file_names:
                    file_count += 1
                    total_source_file_count += 1
                    file_path = os.path.join(dir_path, file_name)
                    # print("file path : ", file_path)
                    path_before_ext, file_ext = os.path.splitext(file_path)

                    if file_ext in filter_extensions:
                        # print("filtered file : ", file_path)
                        path_splits = file_path.split(os.sep)
                        # print(path_splits)
                        file_name = path_splits[-1]

                        dir_filter_out = execute_path_split_filter(path_splits, path_split_filters)

                        if dir_filter_out:
                            filtered_source_file_count += 1
                            id, ext = file_name.split(".")
                            if id in filtered_ids:
                                src_path = file_path
                                dest_dir = root_dir + copy_dir + "\\{}_grouped_data\\".format(group) + "{}-{}".format(group, category)
                                # print(dest_dir)
                                pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

                                image_id_match_condition = rename_df["image_id"] == int(id)

                                groups = list(grouped_ids.keys())
                                rename_params = []
                                rename_params = rename_df.loc[image_id_match_condition, groups].values[0]
                                new_file_name = id
                                for i, param in enumerate(rename_params):
                                    new_file_name += "_" + groups[i] + "-" + str(param)
                                dest_path = dest_dir + "\\" + new_file_name + ".{}".format(ext)
                                # print(dest_path)

                                if not (os.path.exists(dest_path)):
                                    if file_count < 10:
                                        print("SRC : ", src_path)
                                        print("DEST : ", dest_path)
                                    if not test_mode:
                                        shutil.copy2(src_path, dest_path)
                                        print("Copied")
                                        total_copy_count += 1
                                else:
                                    print("{} File already exists. So not copied.")

                if verbose:
                    if file_count < 10:
                        print_file_details(dir_path, file_name)
                    else:
                        continue

    print("All relevant files are copied successfully into respective folders")
    print("total source files count : ", total_source_file_count)
    print("total filtered source files count : ", filtered_source_file_count)
    print("total_copy_count : ", total_copy_count)


def create_combined_all_class_dataset(root_dir, rename_df, grouped_by=["speed", "angle"], data_dirs=["new_data_part_3_grouped"],
                                      copy_dir="\\data", path_split_filters=[(-2,"training_data"), (-3,"training_data")],
                                      ignore_dirs=["^\..*", "corrupted files"], ignore_files=["3884"],
                                      filter_extensions=[".png", ".jpg"], test_mode=True, verbose=False):
    total_file_count = 0
    total_copy_count = 0

    for dir_path, sub_dirs, file_names in os.walk(root_dir):

        if verbose:
            print_oswalk_details(dir_path, sub_dirs, file_names)

        dir_name = os.path.basename(dir_path)

        # ----------- filtering out directories that match the ignore_dirs parameter --------------
        if any(re.match(pattern, dir_name) for pattern in ignore_dirs):
            print("Skipping directory ", dir_path)
            sub_dirs.clear()
            print("skipping all sub directories - ", sub_dirs)
            continue

        # ------------ filtering only the directries in data_dir parameter -------------------------
        for dir in data_dirs:
            if dir_name != dir:
                print("Checking if its a data directory")
                print("Skipping directory ", dir_path)
                continue

        file_count = 0
        for file_name in file_names:

            file_path = os.path.join(dir_path, file_name)
            # print("file path : ", file_path)
            path_before_ext, file_ext = os.path.splitext(file_path)

            if file_ext in filter_extensions:
                # print("filtered file : ", file_path)
                path_splits = file_path.split(os.sep)
                # print(path_splits)
                file_name = path_splits[-1]
                file_name_without_ext = ".".join(file_name.split(".")[:-1])
                # print(file_name_without_ext)
                dir_filter_out = execute_path_split_filter(path_splits, path_split_filters)

                if dir_filter_out and file_name_without_ext not in ignore_files:

                    ##-------- filtered files ---------------------
                    total_file_count += 1
                    ext = file_name.split(".")[-1]
                    id = file_name_without_ext.split(".")[0]
                    image_labels = "_".join(file_name_without_ext.split("_")[1:])
                    # print(image_labels)

                    src_path = file_path
                    dest_dir = root_dir + copy_dir + "\\"
                    for group in grouped_by:
                        dest_dir += "{}_".format(group)
                    dest_dir += "grouped_data\\"

                    if rename_df:
                        image_id_match_condition = rename_df["image_id"] == int(id)

                        rename_params = []
                        rename_params = rename_df.loc[image_id_match_condition, grouped_by].values[0]
                        new_file_name = id
                        for i, param in enumerate(rename_params):
                            dest_dir += grouped_by[i] + "-" + str(param)
                            if i < len(rename_params)-1:
                                dest_dir += "_"
                            new_file_name += "_" + grouped_by[i] + "-" + str(param)
                    else:
                        dest_dir += "{}".format(image_labels)
                        new_file_name = file_name_without_ext

                    # print(dest_dir)
                    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

                    dest_path = dest_dir + "\\" + new_file_name + ".{}".format(ext)
                    # print(dest_path)

                    if not (os.path.exists(dest_path)):
                        if file_count < 10:
                            print("SRC : ", src_path)
                            print("DEST : ", dest_path)
                        if not test_mode:
                            total_copy_count += 1
                            shutil.copy2(src_path, dest_path)
                            print("Copied")
                    else:
                        print("{} File already exists. So not copied.")

            if verbose:
                if file_count < 10:
                    print_file_details(dir_path, file_name)
                else:
                    continue

    print("File Count : {}".format(total_file_count))
    print("Copy Count : {}".format(total_copy_count))
    print("All relevant files are renamed and copied successfully into respective class folders")


def get_csv_from_file_names(dir_path,
                            file_name_column_map={"image_id": [("_",0)], "speed": [("_",1),("-",-1)], "angle": [("_",2),("-",-1)]},
                            csv_path="/data/new_cleaned_data_training_norm.csv", test_mode=True):

    df = pd.DataFrame()
    files = os.listdir(dir_path)
    print("Number of files: ", len(files))

    for i, file_name in enumerate(files):
        ext = file_name.split(".")[-1]
        file_name_without_ext = ".".join(file_name.split(".")[:-1])

        for key,val in file_name_column_map.items():
            column_val = file_name_without_ext
            for filter in val:
                # print(filter)
                column_val = column_val.split(filter[0])[filter[1]]
            df.loc[i,key] = column_val
            # print(i, key, column_val)

    if not test_mode:
        df.to_csv(csv_path, index=False)

    return df





##------------------- Modelling  -------------------------------------------------------------------------------------------------

def create_CNN_model(input_shape, hidden_layers, pretrained_model=None, num_non_trainable_layers=1,
                     output_layer={'BC': [1, 'sigmoid', 'binary_crossentropy'],
                                   'MC': [17, 'softmax', 'categorical_crossentropy']},
                     init='normal', optimize='adam', metrics=['accuracy', 'mse']):
    if pretrained_model:
        if pretrained_model == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'vgg19':
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'inceptionv3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'inceptionresnetv2':
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'xception':
            base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'mobilenet':
            base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'densenet':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'nasnet':
            base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
        elif pretrained_model == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError('Invalid pretrain_model parameter. Please select from vgg16, vgg19, resnet50, '
                             'inceptionv3, inceptionresnetv2, xception, mobilenet, densenet, nasnet, or efficientnet')

        # Set layers to be non-trainable
        for layer in base_model.layers[:-num_non_trainable_layers]:
            layer.trainable = False

        # create model
        model = Sequential()
        model.add(base_model)

    else:
        # create model
        model = Sequential()

    # print(hidden_layers)
    k = 3
    s = 1
    p = 2
    r = 0.2

    for i, layer in enumerate(hidden_layers):
        #         print(layer)
        #         print(type(layer))
        layer = str(layer)
        if i == 0:
            if '_' in layer:
                params = layer.split('_')
                for i, param in enumerate(params):
                    if i == 0:
                        filters = int("".join(filter(str.isdigit, param)))
                    #                     print(filters)
                    elif i == 1:
                        k = int("".join(filter(str.isdigit, param)))
                    #                     print(k)
                    elif i == 2:
                        s = int("".join(filter(str.isdigit, param)))
            #                     print(s)
            else:
                filters = int("".join(filter(str.isdigit, layer)))
            #               print(filters)

            model.add(Conv2D(filters, kernel_size=(k, k), strides=(s, s), activation='relu', input_shape=input_shape))

        elif 'C' in layer:
            if '_' in layer:
                params = layer.split('_')
                for i, param in enumerate(params):
                    if i == 0:
                        filters = int("".join(filter(str.isdigit, param)))
                    #                     print(filters)
                    elif i == 1:
                        k = int("".join(filter(str.isdigit, param)))
                    #                     print(k)
                    elif i == 2:
                        s = int("".join(filter(str.isdigit, param)))
            #                     print(s)
            else:
                filters = int("".join(filter(str.isdigit, layer)))
            #               print(filters)

            model.add(Conv2D(filters, (k, k), strides=(s, s), activation='relu'))


        elif 'MP' in layer:
            stride_flag = False
            if '_' in layer:
                params = layer.split('_')
                for i, param in enumerate(params):
                    if i == 1:
                        p = int("".join(filter(str.isdigit, param)))
                    #                     print(p)
                    elif i == 2:
                        s = int("".join(filter(str.isdigit, param)))
                        stride_flag = True
            #                     print(s)

            model.add(MaxPooling2D(pool_size=(p, p), strides=(s, s) if stride_flag else None))

        elif 'BN' in layer:

            model.add(BatchNormalization())


        elif 'F' in layer:

            model.add(Flatten())
        #             print('F')

        elif 'D' in layer:
            if '_' in layer:
                params = layer.split('_')
                for i, param in enumerate(params):
                    if i == 1:
                        r = int("".join(filter(str.isdigit, param)))

            model.add(Dropout(r))

        else:

            model.add(Dense(int(layer), kernel_initializer=init, activation='relu'))

    loss_functions = []
    for layer, params in output_layer.items():
        model.add(Dense(params[0], activation=params[1], kernel_initializer=init))
        loss_functions.append(params[2])

    model.compile(loss=loss_functions, optimizer=optimize, metrics=metrics)

    return model


def get_ML_model(model_config_path, model_weights_path, _loss_=['binary_crossentropy', 'categorical_crossentropy'], _optimizer_='adam', _metrics_='accuracy'):

    # load YAML and create model
    yaml_file = open(model_config_path, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    loaded_model = model_from_yaml(loaded_model_yaml)
    print("Model loaded")

    # load weights into new model
    loaded_model.load_weights(model_weights_path)
    print("Model_weights_loaded")

    # evaluate loaded model on test data
    loaded_model.compile(loss=_loss_, optimizer=_optimizer_, metrics=[_metrics_])
    print("Model compiled")

    return loaded_model

#-----------------------------------------------------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes, cnf_save_path,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cnf_fig = plt.figure(figsize=(16, 6), dpi=500)

    normalize = False
    plot_name = "Confusion matrix, without normalization"
    print(plot_name)

    ax0 = cnf_fig.add_subplot(121)
    cax = ax0.imshow(cm, interpolation='nearest', cmap=cmap)
    ax0.set_title(plot_name, fontweight="bold", size=16)
    plt.colorbar(cax)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    ax0.set_ylabel('True label', fontsize=16)
    ax0.set_xlabel('Predicted label', fontsize=16)
    # -------------------------------------------------------------------
    normalize = True
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_name = "Normalized confusion matrix"
    print(plot_name)

    ax1 = cnf_fig.add_subplot(122)
    cax = ax1.imshow(cm, interpolation='nearest', cmap=cmap)
    ax1.set_title(plot_name, fontweight="bold", size=16)
    plt.colorbar(cax)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    ax1.set_ylabel('True label', fontsize=16)
    ax1.set_xlabel('Predicted label', fontsize=16)
    plt.savefig(cnf_save_path)
    plt.show()

    print(cm)

#-----------------------------------------------------------------------------------------------------------------------

def plot_histories(histories, plot_save_path):
    plt.clf()
    loss = []
    val_loss = []
    acc = []
    val_acc = []
    for history in histories:
        for error in history['loss']:
            loss.append(error)
        for error in history['acc']:
            acc.append(error)
        for error in history['val_loss']:
            val_loss.append(error)
        for error in history['val_acc']:
            val_acc.append(error)
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'b', label='loss')
    plt.plot(epochs, val_loss, 'c', label='val_loss')
    plt.title('Model Loss Statistics')
    plt.legend()
    plt.show()

    plt.plot(epochs, acc, 'g', label='acc')
    plt.plot(epochs, val_acc, 'r', label='val_acc')
    plt.title('Model Accuracy Statistics')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(16, 6), dpi=500)
    ax0 = fig.add_subplot(121)
    ax0.plot(epochs, loss, 'b', label='loss')
    ax0.plot(epochs, val_loss, 'c', label='val_loss')
    ax0.set_title('Model Loss Statistics')
    ax0.legend()
    ax1 = fig.add_subplot(122)
    ax1.plot(epochs, acc, 'g', label='acc')
    ax1.plot(epochs, val_acc, 'r', label='val_acc')
    ax1.set_title('Model Accuracy Statistics')
    ax1.legend()
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------

def plot_ROC_curves(y_test, y_pred, n_classes, roc_plot_save_path_full, roc_plot_save_path_zoom):
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(roc_plot_save_path_full, dpi=1000)
    plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.5)
    plt.ylim(0.5, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(roc_plot_save_path_zoom, dpi=1000)
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

#-----------------------------------------------------------------------------------------------------------------------------

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

#-----------------------------------------------------------------------------------------------------------------------------

def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))

#-----------------------------------------------------------------------------------------------------------------------------

def plot_classification_report(cr, report_save_path, title='Classification report', with_avg_total=True, _cmap='RdBu'):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    print(lines)
    lines = list(filter(None, lines))
    for line in lines:
        print(line)
        print("---")

    for line in lines[1: -1]:
        print(line)
        t = line.strip().split()
        print(t)
        if len(t) <= 3: continue
        #         print(t)

        v = [float(x) for x in t[-4: -1]]
        support.append(int(t[-1]))
        class_names.append(" ".join(t[:-4]))
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[-1].strip().split()
        print("aveTotal : ", aveTotal)
        vAveTotal = [float(x) for x in aveTotal[-4:-1]]
        support.append(int(aveTotal[-1]))
        class_names.append(" ".join(aveTotal[:-4]))
        plotMat.append(vAveTotal)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    print(['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)])

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=_cmap)
    plt.savefig(report_save_path, dpi=500, format='png', bbox_inches='tight')
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------

def pandas_classification_report(y_true, y_pred, class_names, float_digits=4):

    from sklearn.metrics import precision_recall_fscore_support

    metrics_summary = precision_recall_fscore_support(y_true, y_pred)

    avg = list(precision_recall_fscore_support(y_true, y_pred, average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']

    class_report_df = pd.DataFrame(list(metrics_summary), index=metrics_sum_index, columns=class_names)

    #     avg = (class_report_df.loc[metrics_sum_index[:-1]] * class_report_df.loc[metrics_sum_index[-1]]).sum(axis=1) / total

    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T

