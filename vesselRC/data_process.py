#!/usr/bin/env python3
# Based on: https://github.com/facebookresearch/DeepSDF using MIT LICENSE (https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE)
# Copyright 2021-present Philipp Friedrich, Josef Kamysek. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
from mesh_to_sdf import sample_sdf_near_surface
import mesh_to_sdf
import numpy as np
import trimesh

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


def process_mesh(mesh_filepath, target_filepath):
    print(mesh_filepath + " --> " + target_filepath)
    mesh = trimesh.load(mesh_filepath)
    points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)
    sdf_points= np.concatenate([points,sdf.reshape(-1,1)], axis=1)
    mesh = trimesh.load(mesh_filepath)
    mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
    t = mesh_to_sdf.get_surface_point_cloud(mesh, sample_point_count=100000, calculate_normals=True)
    np.savez(target_filepath, sdf_points = sdf_points,point_cloud = t.points)
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

    args = arg_parser.parse_args()

    additional_general_args = []

    # -> os.path.normpath 用于normal path with remove like '//'
    # -> os.path.basename return the filename without the path
    
    # [F] target path
    dest_dir = args.data_dir

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    # -> source_name is the shapenetCore.v2
    #append_data_source_map(args.data_dir, args.source_name, args.source_dir)

    #class_directories = split[args.source_name]
    class_directories = os.listdir(args.source_dir)
    meshes_targets_and_specific_args = []
    extension ='.npz'
    
    for class_dir in class_directories: #../IntrA/{annotated、complete...}
        class_path = os.path.join(args.source_dir, class_dir)
        shape_dir = os.path.join(args.source_dir, class_dir)

        target_dir = os.path.join(dest_dir, class_dir)

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        mesh_filenames = os.listdir(shape_dir)

        for mesh_filename in mesh_filenames:
            if '.obj' in mesh_filename:
                processed_filepath = os.path.join(target_dir, mesh_filename).replace('.obj',extension)
            else:
                continue
            if args.skip and os.path.isfile(processed_filepath):
                print("skipping " + processed_filepath)
                continue

            meshes_targets_and_specific_args.append(
                (
                    os.path.join(shape_dir, mesh_filename),
                    processed_filepath
                )
            )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:

        for (
            mesh_filepath,
            target_filepath
        ) in meshes_targets_and_specific_args:
            process_mesh(mesh_filepath, target_filepath)
            #print(specific_args+ additional_general_args)
            # executor.submit(
            #     process_mesh,# -> the function， the next arg was for it.
            #     mesh_filepath,
            #     target_filepath
            # )

        # executor.shutdown()
