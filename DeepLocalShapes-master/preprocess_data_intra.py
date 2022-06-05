#!/usr/bin/env python3
# Based on: https://github.com/facebookresearch/DeepSDF using MIT LICENSE (https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE)
# Copyright 2021-present Philipp Friedrich, Josef Kamysek. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess

import deep_ls
import deep_ls.workspace as ws


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:

        passed_classes = passed_classes.union(
            set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes))
        )

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(set(filter(regex.match, classes)))

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh(mesh_filepath, target_filepath, executable, additional_args):
    print(mesh_filepath + " --> " + target_filepath)
    command = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc.wait()


def append_data_source_map(data_dir, name, source):

    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to "
        + "a dataset.",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samplies for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for evaluation. "
        + "Otherwise, the script will produce SDF samples for training.",
    )

    deep_ls.add_common_args(arg_parser) # -> add other args

    args = arg_parser.parse_args()

    additional_general_args = []
    
    # [F] get the bin executable
    deepls_dir = os.path.dirname(os.path.abspath(__file__))
    if args.surface_sampling:
        executable = os.path.join(deepls_dir, "bin/SampleVisibleMeshSurface")
        subdir = ws.surface_samples_subdir
        extension = ".ply"
    else:
        executable = os.path.join(deepls_dir, "bin/PreprocessMesh")
        subdir = ws.sdf_samples_subdir # -> "SurfaceSamples"
        extension = ".npz"
        # -> the extension is for the  processed data
        if args.test_sampling:
            additional_general_args += ["-t"]

    # -> os.path.normpath 用于normal path with remove like '//'
    # -> os.path.basename return the filename without the path
    
    # [F] target path
    dest_dir = os.path.join(args.data_dir, subdir)

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    if args.surface_sampling:
        normalization_param_dir = os.path.join(
            args.data_dir, ws.normalization_param_subdir, args.source_name
        )
        if not os.path.isdir(normalization_param_dir):
            os.makedirs(normalization_param_dir)
    # -> source_name is the shapenetCore.v2
    #append_data_source_map(args.data_dir, args.source_name, args.source_dir)

    #class_directories = split[args.source_name]
    class_directories = os.listdir(args.source_dir)
    meshes_targets_and_specific_args = []

    for class_dir in class_directories: #class
        class_path = os.path.join(args.source_dir, class_dir)
        shape_dir = os.path.join(args.source_dir, class_dir)

        target_dir = os.path.join(dest_dir, class_dir)

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        mesh_filenames = os.listdir(shape_dir)
        specific_args = []

        for mesh_filename in mesh_filenames:
            processed_filepath = os.path.join(target_dir, mesh_filename).replace('.obj',extension)
            if args.skip and os.path.isfile(processed_filepath):
                print("skipping " + processed_filepath)
                continue
            else:
                print("processing " + mesh_filename)
            if args.surface_sampling:
                normalization_param_target_dir = os.path.join(
                    normalization_param_dir, class_dir
                )

                if not os.path.isdir(normalization_param_target_dir):
                    os.mkdir(normalization_param_target_dir)

                normalization_param_filename = os.path.join(
                    normalization_param_target_dir, instance_dir + ".npz"
                )
                specific_args = ["-n", normalization_param_filename]

            meshes_targets_and_specific_args.append(
                (
                    os.path.join(shape_dir, mesh_filename),
                    processed_filepath,
                    specific_args,
                )
            )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:

        for (
            mesh_filepath,
            target_filepath,
            specific_args,
        ) in meshes_targets_and_specific_args:
            #print(specific_args+ additional_general_args)
            executor.submit(
                process_mesh,# -> the function， the next arg was for it.
                mesh_filepath,
                target_filepath,
                executable,
                specific_args + additional_general_args,
            )

        executor.shutdown()
