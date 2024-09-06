#!/usr/bin/env python
from __future__ import annotations

import os
import glob

import cv2
import numpy as np
from tqdm import tqdm
import open3d as o3d
import torch

import sys

sys.path.append("..")
np.random.seed(42)

def map_to_pixels(point3d,w,h,projection,view, z_near=10, z_far=2000):
    # https://stackoverflow.com/questions/67517809/mapping-3d-vertex-to-pixel-using-pyreder-pyglet-opengl
    p=projection@np.linalg.inv(view)@point3d
    p=p/p[3]
    p[0]=(w/2*p[0]+w/2)    #tranformation from [-1,1] ->[0,width]
    p[1]= (h-(h/2*p[1]+h/2))  #tranformation from [-1,1] ->[0,height] (top-left image)
    p[2] = -2 * z_near * z_far / ((z_far - z_near) * p[2] - z_near - z_far)
    return p.T[:, :3].astype(int)

def get_visible_points_view(points, poses, depth_images, intrinsics, vis_threshold = 2.5, color_images = None):
    # Initialization
    X = np.append(points, np.ones((len(points),1)), axis = -1) 
    n_points = X.shape[0] 
    resolution = depth_images[0].shape
    height = resolution[0]  
    width = resolution[1]
    
    # fx = 1169.621094
    # fy = 1167.105103
    # cx = 646.295044 
    # cy = 489.927032
    # intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # intrinsics = np.array([[ 3.75, 0.,  -0.875,  0. ],
    #                        [  0.,   5.,   1.5,  0.],
    #                          [  0.,   0.,  -1.01005025, -20.10050251],
    #                          [  0.,   0.,  -1.,   0.]], dtype=np.float32)

    img2points = {}
    # points2img = {}
    # projected_points = np.zeros((n_points, 2), dtype = int)
    print(f"[INFO] Computing the visible points in each view.")
    
    for i in tqdm(range(len(poses)), desc="Projecting points to views"): # for each view
        # *******************************************************************************************************************
        # STEP 1: get the projected points
        # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
        #projected_points_not_norm = (intrinsics @ np.linalg.inv(poses[i]) @ X.T).T
        # pose = np.linalg.inv(poses[i])[:3,:] # np.concatenate([poses[i][:3,:3].T, -poses[i][:3, 3:4]], axis=1)
        # projected_points_not_norm = (intrinsics @ pose @ X.T).T
        projected_points = map_to_pixels(X.T, width, height, intrinsics, poses[i])
        # Get the mask of the points which have a non-null third coordinate to avoid division by zero
        # mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
        # # Get non homogeneous coordinates of valid points (2D in the image)
        # projected_points[mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
        #         projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
        
        # *******************************************************************************************************************
        # STEP 2: occlusions computation
        # Load the depth from the sensor
        inside_mask = (projected_points[:,0] >= 0) * (projected_points[:,1] >= 0) \
                            * (projected_points[:,0] < width) \
                            * (projected_points[:,1] < height)
        pi = projected_points.T
        # Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
        # point_depth = projected_points_not_norm[:,2]
        point_depth = projected_points[:,2]
        # Compute the visibility mask, true for all the points which are visible from the i-th view
        # @TODO shouldn't we mask the points with negative depth?
        visibility_mask = (np.abs(depth_images[i][pi[1][inside_mask], pi[0][inside_mask]]
                                    - point_depth[inside_mask]) <= \
                                    vis_threshold).astype(bool)
        
        inside_mask[inside_mask == True] = visibility_mask
        inside_points = projected_points[inside_mask]
        if color_images is not None and i % 10 == 0:
            import matplotlib.pyplot as plt
            # plt.subplot(1,3,1)
            # plt.scatter(projected_points[mask][:,0], projected_points[mask][:,1], c=point_depth[mask])
            # plt.colorbar()     
            error = np.abs(depth_images[i][pi[1][inside_mask], pi[0][inside_mask]]
                                    - point_depth[inside_mask])  
            # plt.subplot(2,2,1)
            # plt.scatter(inside_points[:,0], -inside_points[:,1], c=error)
            # plt.colorbar()
            blurred = cv2.blur(color_images[i], (5,5))  
            colors = [np.array(blurred).astype(float)[position[1], position[0]] / 255 for position in inside_points] 
            plt.subplot(2,2,1)
            plt.scatter(inside_points[:,0], -inside_points[:,1], c=colors, s=5)
            # fix aspect ratio
            plt.gca().set_aspect('equal', adjustable='box')
            plt.subplot(2,2,2)
            plt.scatter(inside_points[:,0], -inside_points[:,1], c=point_depth[inside_mask], s=5)
            plt.colorbar()
            # fix aspect ratio
            plt.gca().set_aspect('equal', adjustable='box')
            plt.subplot(2,2,3)
            plt.imshow(depth_images[i])
            plt.colorbar()
            plt.subplot(2,2,4)
            plt.imshow(color_images[i])
            plt.savefig(f"tmp_plot_{i}.png")
            # clear
            plt.clf()
        # if inside_mask.sum() > 0:
        #     print(inside_points.max(),  inside_points.min())
        img2points[i] = {"indices": np.arange(0, len(projected_points))[inside_mask], "projected_points": inside_points}
        
        # print(f"Image {i} has {img2points[i]['indices'].shape[0]} visible points: {img2points[i]['indices']}")
        # for point_idx in img2points[i]["indices"]:
        #     if point_idx in points2img:
        #         points2img[point_idx]["indices"].append(i)
        #         points2img[point_idx]["projected_points"].append(projected_points[point_idx])
        #     else:
        #         points2img[point_idx] = {"indices": [i], "projected_points": [projected_points[point_idx]]}
            # print("Point", point_idx, "is visible in view", i)

    # for point_idx in points2img:
    #     points2img[point_idx]["indices"] = np.array(points2img[point_idx]["indices"])
    #     points2img[point_idx]["projected_points"] = np.array(points2img[point_idx]["projected_points"])

    #@TODO dict is not a scalable enough data structure it seems
    print(f"[INFO] Total number of points: {n_points}")
    print("[INFO] Points per view")
    print(f"[INFO] Average number of points per view: {sum(img2points[i]['indices'].shape[0] for i in img2points) / len(img2points)}")
    print(f"[INFO] Maximum number of points per view: {np.max([img2points[i]['indices'].shape[0] for i in img2points])}")
    print(f"[INFO] Minimum number of points per view: {np.min([img2points[i]['indices'].shape[0] for i in img2points])}")
    print(f"[INFO] Median number of points per view: {np.median([img2points[i]['indices'].shape[0] for i in img2points])}")
    # print("[INFO] Views per point")
    # print(f"[INFO] Average number of views per point: {sum(points2img[i]['indices'].shape[0] for i in points2img) / max(1,len(points2img))}")
    # print(f"[INFO] Maximum number of views per point: {np.max([points2img[i]['indices'].shape[0] for i in points2img])}")
    # print(f"[INFO] Minimum number of views per point: {np.min([points2img[i]['indices'].shape[0] for i in points2img])}")
    # print(f"[INFO] Median number of views per point: {np.median([points2img[i]['indices'].shape[0] for i in points2img])}")
    return img2points


def project_features_to_pixel(mask_features, image_masks, image_features = None, n_occurrences = None):
    """
    @TODO torchify
    """
    if image_features is None:
        image_features = np.zeros((image_masks.shape[0], image_masks.shape[1], image_masks.shape[2], mask_features.shape[1]))
    else:
        image_features[:] = 0
    if n_occurrences is None:
        n_occurrences = np.zeros((image_masks.shape[0], image_masks.shape[1], image_masks.shape[2]))
    else:
        n_occurrences[:] = 0
    # mask_sizes = []
    # @TODO if/else for few masks? Batchify?
    for mask_ix in range(len(mask_features)):
        is_inside_mask = image_masks == mask_ix
        n_occurrences += is_inside_mask
        image_features[is_inside_mask] += mask_features[mask_ix]
        # for level in range(len(image_masks)):
        #     mask_sizes.append((image_masks[level] == mask_ix).sum())
        # for level in range(len(image_masks)):
        #     is_inside_mask = image_masks[level] == mask_ix
        #     n_occurrences[level] += is_inside_mask
        #     image_features[level][is_inside_mask] += mask_features[mask_ix]
    # mask_sizes = np.array(mask_sizes) / (image_masks.shape[1] * image_masks.shape[2])
    # print(np.histogram(mask_sizes, bins=10))
    image_features /= np.maximum(1, n_occurrences[..., np.newaxis])
    return image_features, n_occurrences

def project_features_to_pixel_torch(mask_features, image_masks, image_features = None, n_occurrences = None):
    """exactly the same in torch"""
    if image_features is None:
        image_features = torch.zeros((image_masks.shape[0], image_masks.shape[1], image_masks.shape[2], mask_features.shape[1]))
    else:
        image_features[:] = 0
    if n_occurrences is None:
        n_occurrences = torch.zeros((image_masks.shape[0], image_masks.shape[1], image_masks.shape[2]))
    else:
        n_occurrences[:] = 0
    for mask_ix in range(len(mask_features)):
        is_inside_mask = image_masks == mask_ix
        n_occurrences += is_inside_mask
        image_features[is_inside_mask] += mask_features[mask_ix]

    is_inside_mask2 = image_masks[..., None] == torch.arange(mask_features.shape[0])[None, None, None, :]
    n_occurrences2 = is_inside_mask2.sum(dim=(3,)) 
    image_features2 = torch.zeros((image_masks.shape[0], image_masks.shape[1], image_masks.shape[2], mask_features.shape[1]))
    image_features2[is_inside_mask2] = mask_features
    image_features2 /= torch.clamp(n_occurrences2[..., None], min=1)
    assert torch.allclose(image_features, image_features2)

    image_features /= torch.clamp(n_occurrences[..., None], min=1)
    return image_features, n_occurrences


def projected_point_to_pixel(projected_points, resolution):
    height, width = resolution
    projected_points = projected_points.astype(int)
    projected_points = projected_points[(projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)]
    return projected_points

def extract_averaged_point_features(pixel_wise_features, points2img):
    print("[INFO] Extracting averaged point features..")
    averaged_features = {}
    for point_idx in tqdm(points2img, desc="Extracting averaged point features"):
        averaged_features[point_idx] = np.mean(pixel_wise_features[points2img[point_idx]["indices"]], axis = 0)
    return averaged_features


def load_data(obj_path, images_path, depth_path, feat_path, mask_path, poses_path, intrinsics_path, full_embedding_path = None, full_embeddings_mode = False, max_points=int(1*1e6)):
    
    print("[INFO] Loading the data..")
    # mesh = o3d.io.read_point_cloud(ply_path)
    mesh = o3d.io.read_triangle_model(obj_path)
    try:
        points3D = mesh.points
    except:
        try:
            points3D = np.concatenate([mesh.meshes[i].mesh.vertices for i in range(len(mesh.meshes))])
        except ValueError:
            mesh =  o3d.io.read_triangle_mesh(obj_path)
            points3D = mesh.vertices
    if len(points3D) > max_points:
        # randomly subsample points
        points3D = points3D[np.random.choice(len(points3D), max_points, replace=False)]



    # if mask_path == feat_path:
    #     print("[INFO] Mask and feature paths are the same. Selecting based on _s, _f endings")
    #     masks = [np.load(path) for path in tqdm(sorted(glob.glob(os.path.join(mask_path, "*_s.npy"))), desc="Reloading masks")]
    #     features = [np.load(path).astype(np.float16) for path in tqdm(sorted(glob.glob(os.path.join(feat_path, "*_f.npy"))), desc="Reloading features")]

    if full_embeddings_mode:
        if full_embedding_path is None:
            raise ValueError("Full embeddings mode is enabled but no full embedding path is provided")
        full_embeddings = np.load(os.path.join(full_embedding_path, "embeddings.npy"))
        images = [cv2.imread(path) for path in tqdm(sorted(glob.glob(os.path.join(images_path, "*.jpg"))), desc="Loading images")]
        masks = [np.zeros((1,) + i.shape[:2], dtype=int) for i in images]
        features = [np.expand_dims(full_embeddings[i], 0).astype(np.float16) for i in range(len(images))] 
        depth_images = [np.load(path).astype(np.float16) for path in tqdm(sorted(glob.glob(os.path.join(depth_path, "*.npy"))), desc="Loading depth images")]
        # features = [np.load(path).astype(np.float16) for path in tqdm(sorted(glob.glob(os.path.join(feat_path, "*.npy"))), desc="Loading features")]
        # masks = [np.load(path) for path in tqdm(sorted(glob.glob(os.path.join(mask_path, "*.npy"))), desc="Loading masks")]
        poses = [np.loadtxt(path) for path in tqdm(sorted(glob.glob(os.path.join(poses_path, "*.txt"))), desc="Loading poses")]
        assert len(images) == len(depth_images) == len(features) == len(masks) == len(poses)
    else: # True or not len(images) == len(depth_images) == len(features) == len(masks) == len(poses):
        # print("Warning: Number of images, depth images, features, masks and poses do not match")
        # if full_embeddings_mode:
        #     raise ValueError("Full embeddings mode is not supported when the number of images, depth images, features, masks and poses do not match")
        # only keep the ones that have all the data: match based on names
        image_names = [os.path.basename(path).split(".")[0] for path in sorted(glob.glob(os.path.join(images_path, "*.jpg")))]
        depth_image_names = [os.path.basename(path).split(".")[0] for path in sorted(glob.glob(os.path.join(depth_path, "*.npy")))]
        feature_names = [os.path.basename(path).split(("." if feat_path != mask_path else "_f."))[0] for path in sorted(glob.glob(os.path.join(feat_path, "*.npy")))]
        mask_names = [os.path.basename(path).split(("." if feat_path != mask_path else "_s."))[0] for path in sorted(glob.glob(os.path.join(mask_path, "*.npy")))]
        pose_names = [os.path.basename(path).split(".")[0] for path in sorted(glob.glob(os.path.join(poses_path, "*.txt")))]
        common_names = set(image_names) & set(depth_image_names) & set(feature_names) & set(mask_names) & set(pose_names)
        print("[INFO] Loading images, depth images, features, masks and poses based on common names")
        images = [cv2.imread(os.path.join(images_path, name + ".jpg")) for name in common_names]
        depth_images = [np.load(os.path.join(depth_path, name + ".npy")).astype(np.float16) for name in common_names]
        features = [np.load(os.path.join(feat_path, name + (".npy" if feat_path != mask_path else "_f.npy"))).astype(np.float16) for name in common_names]
        masks = [np.load(os.path.join(mask_path, name + (".npy" if feat_path != mask_path else "_s.npy"))) for name in common_names]
        poses = [np.loadtxt(os.path.join(poses_path, name + ".txt")) for name in common_names]
        assert len(images) == len(depth_images) == len(features) == len(masks) == len(poses)
    intrinsics = np.loadtxt(intrinsics_path)

    height, width, channels = images[0].shape

    print(f"[INFO] Number of views: {len(images)}")
    print(f"[INFO] Image dimension: {height}x{width}x{channels}")
    print(f"[INFO] Feature dimension: {features[0].shape}")
    print(f"[INFO] Number of points: {len(points3D)}")
    print(f"[INFO] Mask dimension: {masks[0].shape}")
    print(f"[INFO] Pose dimension: {poses[0].shape}")
    
    n_levels, _, _ = masks[0].shape
    return images, masks, points3D, poses, depth_images, features, intrinsics, n_levels


def convert_to_pcd(obj_path, images_path, depth_path, feat_path, mask_path, poses_path, intrinsics_path, output_path, full_embedding_path = None, full_embeddings_mode = False, max_points=int(1*1e6)):
    # mesh = o3d.io.read_point_cloud(ply_path)
    # mesh = o3d.io.read_triangle_model(obj_path)
    # try:
    #     points3D = mesh.points
    # except:
    #     try:
    #         points3D = np.concatenate([mesh.meshes[i].mesh.vertices for i in range(len(mesh.meshes))])
    #     except ValueError:
    #         mesh =  o3d.io.read_triangle_mesh(obj_path)
    #         points3D = mesh.vertices
    # if len(points3D) > max_points:
    #     # randomly subsample points
    #     points3D = points3D[np.random.choice(len(points3D), max_points, replace=False)]



    # # if mask_path == feat_path:
    # #     print("[INFO] Mask and feature paths are the same. Selecting based on _s, _f endings")
    # #     masks = [np.load(path) for path in tqdm(sorted(glob.glob(os.path.join(mask_path, "*_s.npy"))), desc="Reloading masks")]
    # #     features = [np.load(path).astype(np.float16) for path in tqdm(sorted(glob.glob(os.path.join(feat_path, "*_f.npy"))), desc="Reloading features")]

    # if full_embeddings_mode:
    #     if full_embedding_path is None:
    #         raise ValueError("Full embeddings mode is enabled but no full embedding path is provided")
    #     full_embeddings = np.load(os.path.join(full_embedding_path, "embeddings.npy"))
    #     images = [cv2.imread(path) for path in tqdm(sorted(glob.glob(os.path.join(images_path, "*.jpg"))), desc="Loading images")]
    #     masks = [np.zeros((1,) + i.shape[:2], dtype=int) for i in images]
    #     features = [np.expand_dims(full_embeddings[i], 0).astype(np.float16) for i in range(len(images))] 
    #     depth_images = [np.load(path).astype(np.float16) for path in tqdm(sorted(glob.glob(os.path.join(depth_path, "*.npy"))), desc="Loading depth images")]
    #     # features = [np.load(path).astype(np.float16) for path in tqdm(sorted(glob.glob(os.path.join(feat_path, "*.npy"))), desc="Loading features")]
    #     # masks = [np.load(path) for path in tqdm(sorted(glob.glob(os.path.join(mask_path, "*.npy"))), desc="Loading masks")]
    #     poses = [np.loadtxt(path) for path in tqdm(sorted(glob.glob(os.path.join(poses_path, "*.txt"))), desc="Loading poses")]
    #     assert len(images) == len(depth_images) == len(features) == len(masks) == len(poses)
    # else: # True or not len(images) == len(depth_images) == len(features) == len(masks) == len(poses):
    #     # print("Warning: Number of images, depth images, features, masks and poses do not match")
    #     # if full_embeddings_mode:
    #     #     raise ValueError("Full embeddings mode is not supported when the number of images, depth images, features, masks and poses do not match")
    #     # only keep the ones that have all the data: match based on names
    #     image_names = [os.path.basename(path).split(".")[0] for path in sorted(glob.glob(os.path.join(images_path, "*.jpg")))]
    #     depth_image_names = [os.path.basename(path).split(".")[0] for path in sorted(glob.glob(os.path.join(depth_path, "*.npy")))]
    #     feature_names = [os.path.basename(path).split(("." if feat_path != mask_path else "_f."))[0] for path in sorted(glob.glob(os.path.join(feat_path, "*.npy")))]
    #     mask_names = [os.path.basename(path).split(("." if feat_path != mask_path else "_s."))[0] for path in sorted(glob.glob(os.path.join(mask_path, "*.npy")))]
    #     pose_names = [os.path.basename(path).split(".")[0] for path in sorted(glob.glob(os.path.join(poses_path, "*.txt")))]
    #     common_names = set(image_names) & set(depth_image_names) & set(feature_names) & set(mask_names) & set(pose_names)
    #     print("Loading images, depth images, features, masks and poses based on common names")
    #     images = [cv2.imread(os.path.join(images_path, name + ".jpg")) for name in common_names]
    #     depth_images = [np.load(os.path.join(depth_path, name + ".npy")).astype(np.float16) for name in common_names]
    #     features = [np.load(os.path.join(feat_path, name + (".npy" if feat_path != mask_path else "_f.npy"))).astype(np.float16) for name in common_names]
    #     masks = [np.load(os.path.join(mask_path, name + (".npy" if feat_path != mask_path else "_s.npy"))) for name in common_names]
    #     poses = [np.loadtxt(os.path.join(poses_path, name + ".txt")) for name in common_names]
    #     assert len(images) == len(depth_images) == len(features) == len(masks) == len(poses)
    # intrinsics = np.loadtxt(intrinsics_path)
    
    base_path = os.path.dirname(feat_path)
    print(f"Base Path: {base_path}")

    images, masks, points3D, poses, depth_images, features, intrinsics, n_levels = load_data(obj_path, images_path, depth_path, feat_path, mask_path, poses_path, intrinsics_path, full_embedding_path, full_embeddings_mode, max_points=max_points)

    # for image, depth_image, feature, mask, pose in zip(images, depth_images, features, masks, poses):
    #     # make sure name matches
    #     assert os.path.basename(image).split(".")[0] == os.path.basename(depth_image).split(".")[0] == os.path.basename(feature).split(".")[0] == os.path.basename(mask).split(".")[0] == os.path.basename(pose).split(".")[0]


    # height, width, channels = images[0].shape
    # n_levels, _, _ = masks[0].shape

    # # for image, depth_image, feature, mask, pose in zip(images, depth_images, features, masks, poses):
    # #     # make sure name matches
    # #     assert os.path.basename(image).split(".")[0] == os.path.basename(depth_image).split(".")[0] == os.path.basename(feature).split(".")[0] == os.path.basename(mask).split(".")[0] == os.path.basename(pose).split(".")[0]

    # print(f"[INFO] Number of views: {len(images)}")
    # print(f"[INFO] Image dimension: {height}x{width}x{channels}")
    # print(f"[INFO] Feature dimension: {features[0].shape}")
    # print(f"[INFO] Number of points: {len(points3D)}")
    # print(f"[INFO] Mask dimension: {masks[0].shape}")
    # print(f"[INFO] Pose dimension: {poses[0].shape}")

    img2points = get_visible_points_view(points3D, poses, depth_images, intrinsics)

    
    if False:
        # image_feature_placeholder = np.zeros((n_levels, height, width, features[0].shape[1]))
        # n_occurrences_placeholder = np.zeros((n_levels, height, width))
        # average features
        
        allegiances = [np.expand_dims(np.ones(n_levels) / n_levels, axis=0).repeat(len(features[i]), axis=0) for i in range(len(features))]
        for iter in range(10):
            point_features_sum = np.zeros((n_levels, len(points3D), features[0].shape[1]), dtype=np.float32)
            n_observed = np.zeros((n_levels, len(points3D)), dtype=np.float64)
            # recompute means
            for i in tqdm(range(len(features)), desc="Projecting features to 3D points"):
                # @TODO double gather, don't go via pixel
                # pixel_wise_features, n_occurrences = project_features_to_pixel(features[i], masks[i], image_feature_placeholder, n_occurrences_placeholder)
                # @TODO numpyify or torchify (.gather()?)
                if features[i].shape[-1] != features[0].shape[-1]:
                    continue
                positions = img2points[i]["projected_points"]
                points = img2points[i]["indices"]
                mask_ix = masks[i][:, positions[:, 1], positions[:, 0]]
                
                # point_features_sum[:, points] += allegiances @ (features[i][mask_ix] * (mask_ix != -1)[..., None])
                
                if iter == 0:
                     point_features_sum[:, points] += features[i][mask_ix] * (mask_ix != -1)[..., None]
                     n_observed[:, points] += (mask_ix != -1)
                else:
                    # allegiances[i] has shape (n_masks, n_levels). Here we assign every point the allegiance of the mask it belongs to. 
                    # resulting allegiances has shape (n_levels, n_points, n_levels)
                    alleg = allegiances[i][mask_ix] 
                    # the observed features with shape (n_levels, n_points, n_features)
                    point_features_view = features[i][mask_ix] * (mask_ix != -1)[..., None]
                    # we multiply each hierarchy level by a (n_levels, n_levels) permutation matrix, essentially reassigning the levels
                    point_features_view_reassigned = np.einsum("ino,ink->onk", alleg, point_features_view)
                    point_features_sum[:, points] += point_features_view_reassigned
                    # total_assigned_weight = (mask_ix != -1) * alleg.sum(axis=2)
                    total_assigned_weight = np.einsum("ino,in->on", alleg, (mask_ix != -1))
                    n_observed[:, points] += total_assigned_weight
            point_features_sum /= n_observed[:, :, None] + 1e-6
            # point_features_sum /= point_features_sum.sum(axis=-1, keepdims=True) + 1e-6
            
            overall_similarity_before = 0.0
            overall_similarity_after = 0.0
            overall_masks = 0
            # recompute allegiances
            for i in tqdm(range(len(features)), desc="Recomputing allegiances"):
                if features[i].shape[-1] != features[0].shape[-1]:
                    continue
                positions = img2points[i]["projected_points"]
                points = img2points[i]["indices"]
                mask_ix = masks[i][:, positions[:, 1], positions[:, 0]] # TODO: first compute similarity, then assign
                for mask_id in range(len(features[i])):
                    observed_feature = features[i][mask_id]
                    points_within_mask = points[(mask_ix == mask_id).any(axis=0)]
                    if len(points_within_mask) == 0:
                        continue
                    model_features = point_features_sum[:, points_within_mask]
                    level_similarity = (model_features @ observed_feature).mean(-1)
                    overall_similarity_before += level_similarity @ allegiances[i][mask_id]
                    allegiances[i][mask_id] = np.arange(0, n_levels) == np.argmax(level_similarity, axis=0)
                    overall_similarity_after += level_similarity @ allegiances[i][mask_id]
                    overall_masks += 1
            
            print(f"[INFO] Average similarity before: {overall_similarity_before / overall_masks}")
            print(f"[INFO] Average similarity after: {overall_similarity_after / overall_masks}")
        
        
        point_features_sum = point_features_sum.astype(np.float16)
        print(f"[INFO] Average number of features per point: {np.mean(n_observed)}")
        print(f"[INFO] Maximum number of features per point: {np.max(n_observed)}")
        print(f"[INFO] Minimum number of features per point: {np.min(n_observed)}")
        print(f"[INFO] Median number of features per point: {np.median(n_observed)}")
        print("[INFO] Point feature shape:", point_features_sum.shape)

        base_path = os.path.dirname(feat_path)
        
        if "highlight" in feat_path:
            np.save(f"{base_path}/point_features_highlight.npy", point_features_sum)
            print(f"[INFO] Point features saved to {base_path}/point_features_highlight.npy")
        else:
            np.save(f"{base_path}/point_features_full.npy", point_features_sum)
            print(f"[INFO] Point features saved to {base_path}/point_features_full.npy")
        # if "highlight" in feat_path:
        #     np.save("/home/bieriv/LangSplat/LangSplat/data/rotterdam-output/point_features_langbase.npy", point_features_sum)
        # else:
        #     np.save("/mnt/usb_ssd/opencity-data/openscene-base/point_features_base.npy", point_features_sum)
    if True:
        # average colors
        point_colors_sum = np.zeros((len(points3D), 3), dtype=np.float16)
        n_observed = np.zeros(len(points3D))
        for i in tqdm(range(len(images)), desc="Projecting colors to 3D points"):
            for index, position in zip(img2points[i]["indices"], img2points[i]["projected_points"]):
                point_colors_sum[index] += images[i][position[1], position[0]] / 255.0
                n_observed[index] += 1
        point_colors_sum /= np.maximum(1, n_observed[:, None])
        
        print(np.nanmax(point_colors_sum), np.nanmin(point_colors_sum))

        # create point cloud and save it
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points3D)
        point_cloud.colors = o3d.utility.Vector3dVector(point_colors_sum)
        assert point_cloud.has_points()
        assert point_cloud.has_colors()

        
        o3d.io.write_point_cloud(f"{base_path}/generated_pcd.ply", point_cloud)


if __name__ == "__main__":
    if False:
        base_path = "/home/bieriv/LangSplat/LangSplat/data/brooklyn-bridge/"
        obj_path = "/home/bieriv/LangSplat/LangSplat/data/brooklyn-bridge-obj/brooklyn-bridge.obj"
        full_embeddings_mode = False
    elif True:
        base_path = "/mnt/usb_ssd/opencity-data/data/rotterdam-output/"
        obj_path = "/mnt/usb_ssd/opencity-data/data/rotterdam/rotterdam.obj"
        full_embeddings_mode = False
    elif False:
        base_path = "/home/bieriv/LangSplat/LangSplat/data/eth-output-v1/"
        obj_path = "/home/bieriv/LangSplat/LangSplat/data/eth/eth.glb"
        full_embeddings_mode = False
    elif False:
        base_path = "/home/bieriv/LangSplat/LangSplat/data/ams-output-v2/"
        obj_path = "/home/bieriv/LangSplat/LangSplat/data/ams/ams.glb"
        full_embeddings_mode = False
    elif False:
        base_path = "/home/bieriv/LangSplat/LangSplat/data/delft-output-v1/"
        obj_path = "/home/bieriv/LangSplat/LangSplat/data/delft/delft.glb"
        full_embeddings_mode = False
    else:
        obj_path = "/home/bieriv/LangSplat/LangSplat/data/buenos-aires-squared/buenos-aires-squared-shifted.obj"
        base_path = "/home/bieriv/LangSplat/LangSplat/data/buenos-aires-squared-output-v3/"
        full_embeddings_mode = True
    convert_to_pcd(obj_path = obj_path, #"scene_example_downsampled.ply",
                    images_path= base_path + "color",
                    depth_path = base_path + "depth",
                    feat_path = base_path + "language_features_highlight",
                    mask_path = base_path + "language_features_highlight",
                    full_embedding_path = "/mnt/usb_ssd/results/opencity-data/openscene-base/" + "full_image_embeddings",
                    poses_path = base_path + "pose",
                    intrinsics_path = base_path + "intrinsic/projection_matrix.txt",
                    output_path = "semantic_point_cloud.ply",
                    full_embeddings_mode = full_embeddings_mode,
                    # max_points = 10000
                    )