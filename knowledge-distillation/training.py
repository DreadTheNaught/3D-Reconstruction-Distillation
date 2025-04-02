import sys
import numpy as np
sys.path.append("dust3r/")

from student_model import StudentModel, StudentModelPretrained
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dataloader import get_dataloader
import open3d as o3d

import torch.nn.functional as F

from enum import Enum
import copy
import argparse
import os
import torch
import gc

import logging


class InferenceParams():
    IMAGE_SIZE = 512
    SCENEGRAPH_TYPE = "swin-3"
    DEVICE = "cuda"
    BATCH_SIZE = 8
    GLOBAL_ALIGNMENT_NITER = 300
    SCHEDULE = "linear"
    LEARNING_RATE = 0.01
    FILE_COUNT =  1101

# Initialize teacher and student models
# teacher = TeacherModel()

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--weights_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--scene_type", type=str, default="apt1_kitchen", help="Scene type from 12Scenes dataset")
    return parser


def teacher_inference(args):
    # Load images from both train and test
    scene_dir_train = os.path.join(
        args.dataset_path, args.scene_type, "train", "rgb")
    scene_dir_test = os.path.join(
        args.dataset_path, args.scene_type, "test", "rgb")
    filelist = [os.path.join(scene_dir_train, f)
                for f in os.listdir(scene_dir_train)]
    filelist += [os.path.join(scene_dir_test, f)
                 for f in os.listdir(scene_dir_test)]
    filelist = sorted(filelist, key=lambda x: os.path.basename(x).split('.')[0].split('-')[1])

    # Store the original list before subsetting
    original_filelist = filelist.copy()
    # Apply subsetting
    filelist = filelist[:InferenceParams.FILE_COUNT]

    imgs = load_images(filelist, size=InferenceParams.IMAGE_SIZE)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    # Teacher model teaches...
    model = AsymmetricCroCo3DStereo.from_pretrained(
        args.weights_path).to(InferenceParams.DEVICE)
    pairs = make_pairs(
        imgs, scene_graph=InferenceParams.SCENEGRAPH_TYPE, prefilter=None, symmetrize=False)
    output = inference(pairs, model, InferenceParams.DEVICE,
                       batch_size=InferenceParams.BATCH_SIZE, verbose=True)
    
    del pairs
    torch.cuda.empty_cache()

    mode = GlobalAlignerMode.PointCloudOptimizer if len(
        imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(
        output, device=InferenceParams.DEVICE, mode=mode, verbose=True)
    
    del output
    torch.cuda.empty_cache()
    
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init='mst',
            niter=InferenceParams.GLOBAL_ALIGNMENT_NITER,
            schedule=InferenceParams.SCHEDULE,
            lr=InferenceParams.LEARNING_RATE,
        )
        print(loss)
        pts3D = scene.depth_to_pts3d()
        del model  # Remove the teacher model from GPU memory
        del scene

        # Free up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        pts_dict = {}
        # Track which files were actually processed
        processed_files = set([os.path.basename(f) for f in filelist])

        for i in range(len(filelist)):
            frame_id = filelist[i].split(".")[0]
            ind = int(frame_id.split('-')[1])
            pts_dict[ind] = pts3D[i]

        # Return both the point data and the set of processed files
        return pts_dict, processed_files


def create_dataset_labels(pts3D, processed_files, args):
    """
    Create labels of 3D points only for images that were processed during inference
    """
    pts3d_dir_train = os.path.join(
        args.dataset_path, args.scene_type, "train", "pts3d")
    pts3d_dir_test = os.path.join(
        args.dataset_path, args.scene_type, "test", "pts3d")

    if not os.path.exists(pts3d_dir_train):
        os.makedirs(pts3d_dir_train, exist_ok=True)

    if not os.path.exists(pts3d_dir_test):
        os.makedirs(pts3d_dir_test, exist_ok=True)

    # Handle train directory
    rgb_dir_train = os.listdir(os.path.join(
        args.dataset_path, args.scene_type, "train", "rgb"))
    for f in rgb_dir_train:
        # Only process files that were included in the inference
        if f in processed_files:
            frame_id = f.split(".")[0]
            ind = int(frame_id.split('-')[1])
            if ind in pts3D:
                torch.save(pts3D[ind], os.path.join(
                    pts3d_dir_train, f"{frame_id}.pt"))

    # Handle test directory
    rgb_dir_test = os.listdir(os.path.join(
        args.dataset_path, args.scene_type, "test", "rgb"))
    for f in rgb_dir_test:
        # Only process files that were included in the inference
        if f in processed_files:
            frame_id = f.split(".")[0]
            ind = int(frame_id.split('-')[1])
            if ind in pts3D:
                torch.save(pts3D[ind], os.path.join(
                    pts3d_dir_test, f"{frame_id}.pt"))


def student_learn(student, dataloader, model_type, scene_type, logger, epochs):
    # Use the predicted 3D points to start training
    # breakpoint()
    # student.learn(torch.cat([im['img'] for im in imgs], dim=0).to(InferenceParams.DEVICE), pts3D)

    if not os.path.exists("student_models"):
        os.mkdir("student_models")

    if not os.path.exists("student_models/{}".format(model_type)):
        os.mkdir("student_models/{}".format(model_type))

    best_loss = float("Inf")
    for e in range(epochs):
        i = 0
        for image, label in dataloader:
            i += 1
            loss = student.learn(image, label)
            log_message = f"Epoch: {e}, Iteration: {i}, Loss: {loss}"
            logger.info(log_message)
            if loss < best_loss:
                torch.save(student.state_dict(), "student_models/{}/{}.pth".format(model_type, scene_type))
                best_loss = loss


def create_logger(args):
    train_logger = logging.getLogger('train_logger')
    test_logger = logging.getLogger('test_logger')

    train_logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG
    test_logger.setLevel(logging.DEBUG)

    if not os.path.exists(f'logging/{args.model_type}/{args.scene_type}'):
        if not os.path.exists("logging/{}".format(args.model_type)):
            os.mkdir("logging/{}".format(args.model_type))
        os.mkdir(f'logging/{args.model_type}/{args.scene_type}')

    train_file_handler = logging.FileHandler(f'logging/{args.model_type}/{args.scene_type}/train.log', mode='w')
    train_file_handler.setLevel(logging.DEBUG)  # Set the file handler level to DEBUG
    test_file_handler = logging.FileHandler(f'logging/{args.model_type}/{args.scene_type}/test.log', mode='w')
    test_file_handler.setLevel(logging.DEBUG)  # Set the file handler level to DEBUG

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    train_file_handler.setFormatter(formatter)
    test_file_handler.setFormatter(formatter)
    
    # Add the handlers to the loggers
    train_logger.addHandler(train_file_handler)
    test_logger.addHandler(test_file_handler)

    return train_logger, test_logger

def reduce_pt_size(pts3D):
    """
    Reduce the size of the 3D points to save memory
    """
    for key in pts3D.keys():
        pts3D[key] = pts3D[key].half()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3D[key].cpu().numpy())
        pcd = pcd.voxel_down_sample(voxel_size=0.02)  # Adjust voxel size
        pts3D[key] = torch.tensor(np.asarray(pcd.points), dtype=torch.float16)
        
        
    return pts3D

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train_logger, test_logger = create_logger(args)

    # Get both points and processed files
    pts3D, processed_files = teacher_inference(args)
    create_dataset_labels(pts3D, processed_files, args)

    # Rest of the code remains the same
    train_dataloader = get_dataloader(
        args.dataset_path, batch_size=4, scene_type = args.scene_type,split = "train")
    if args.model_type == 'conv_pretrained':
        student = StudentModelPretrained().to(InferenceParams.DEVICE)
    else:
        student = StudentModel().to(InferenceParams.DEVICE)

    student_learn(student, train_dataloader, args.model_type,
                  args.scene_type, train_logger, epochs=300)

    # Eval
    test_dataloader = get_dataloader(
        args.dataset_path, batch_size=1, scene_type=args.scene_type, split="test")
    for image, label in test_dataloader:
        pred = student(image)
        b, c, _, _ = pred.shape
        pred = torch.transpose(pred.reshape(b, c, -1), 1, 2)
        l2_error = F.mse_loss(pred, label)
        print(l2_error)
        msg = f"Test image - L2 Error: {l2_error}"
        test_logger.info(msg)
