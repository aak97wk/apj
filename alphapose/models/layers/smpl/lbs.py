from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import jittor as jt
from jittor import init
from jittor import nn

import numpy as np

def rot_mat_to_euler(rot_mats):
    sy = jt.sqrt(((rot_mats[:, 0, 0] * rot_mats[:, 0, 0]) + (rot_mats[:, 1, 0] * rot_mats[:, 1, 0])))
    return jt.atan2((- rot_mats[:, 2, 0]), sy)

def find_dynamic_lmk_idx_and_bcoords(vertices, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, neck_kin_chain, dtype=jt.float32):
    ' Compute the faces, barycentric coordinates for the dynamic landmarks\n\n\n        To do so, we first compute the rotation of the neck around the y-axis\n        and then use a pre-computed look-up table to find the faces and the\n        barycentric coordinates that will be used.\n\n        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)\n        for providing the original TensorFlow implementation and for the LUT.\n\n        Parameters\n        ----------\n        vertices: jt.tensor BxVx3, dtype = jt.float32\n            The tensor of input vertices\n        pose: jt.tensor Bx(Jx3), dtype = jt.float32\n            The current pose of the body model\n        dynamic_lmk_faces_idx: jt.tensor L, dtype = jt.long\n            The look-up table from neck rotation to faces\n        dynamic_lmk_b_coords: jt.tensor Lx3, dtype = jt.float32\n            The look-up table from neck rotation to barycentric coordinates\n        neck_kin_chain: list\n            A python list that contains the indices of the joints that form the\n            kinematic chain of the neck.\n        dtype: jt.dtype, optional\n\n        Returns\n        -------\n        dyn_lmk_faces_idx: jt.tensor, dtype = jt.long\n            A tensor of size BxL that contains the indices of the faces that\n            will be used to compute the current dynamic landmarks.\n        dyn_lmk_b_coords: jt.tensor, dtype = jt.float32\n            A tensor of size BxL that contains the indices of the faces that\n            will be used to compute the current dynamic landmarks.\n    '
    batch_size = vertices.shape[0]
    aa_pose = jt.index_select(pose.view((batch_size, (- 1), 3)), 1, neck_kin_chain)
    rot_mats = batch_rodrigues(aa_pose.view((- 1), 3), dtype=dtype).view((batch_size, (- 1), 3, 3))
    rel_rot_mat = jt.eye(3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = jt.bmm(rot_mats[:, idx], rel_rot_mat)
    y_rot_angle = jt.round(jt.clamp((((- rot_mat_to_euler(rel_rot_mat)) * 180.0) / np.pi), max=39)).astype(jt.long)
    neg_mask = y_rot_angle.lt(0).astype(jt.long)
    mask = y_rot_angle.lt((- 39)).astype(jt.long)
    neg_vals = ((mask * 78) + ((1 - mask) * (39 - y_rot_angle)))
    y_rot_angle = ((neg_mask * neg_vals) + ((1 - neg_mask) * y_rot_angle))
    dyn_lmk_faces_idx = jt.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
    dyn_lmk_b_coords = jt.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
    return (dyn_lmk_faces_idx, dyn_lmk_b_coords)

def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ' Calculates landmarks by barycentric interpolation\n\n        Parameters\n        ----------\n        vertices: jt.tensor BxVx3, dtype = jt.float32\n            The tensor of input vertices\n        faces: jt.tensor Fx3, dtype = jt.long\n            The faces of the mesh\n        lmk_faces_idx: jt.tensor L, dtype = jt.long\n            The tensor with the indices of the faces used to calculate the\n            landmarks.\n        lmk_bary_coords: jt.tensor Lx3, dtype = jt.float32\n            The tensor of barycentric coordinates that are used to interpolate\n            the landmarks\n\n        Returns\n        -------\n        landmarks: jt.tensor BxLx3, dtype = jt.float32\n            The coordinates of the landmarks for each mesh in the batch\n    '
    (batch_size, num_verts) = vertices.shape[:2]
    # device = vertices.device
    lmk_faces = faces[lmk_faces_idx].view((- 1)).view((batch_size, (- 1), 3))
    lmk_faces += (jt.arange(batch_size, dtype=jt.long).view(((- 1), 1, 1)) * num_verts)
    lmk_vertices = vertices.view((- 1), 3)[lmk_faces].view((batch_size, (- 1), 3, 3))
    landmarks = jt.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks

def joints2bones(joints, parents):
    ' Decompose joints location to bone length and direction.\n\n        Parameters\n        ----------\n        joints: jt.tensor Bx24x3\n    '
    assert (joints.shape[1] == parents.shape[0])
    bone_dirs = jt.zeros_like(joints)
    bone_lens = jt.zeros_like(joints[:, :, :1])
    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if (p_id == (- 1)):
            bone_dirs[:, c_id] = joints[:, c_id]
        else:
            diff = (joints[:, c_id] - joints[:, p_id])
            length = (jt.norm(diff, dim=1, keepdims=True) + 1e-08)
            direct = (diff / length)
            bone_dirs[:, c_id] = direct
            bone_lens[:, c_id] = length
    return (bone_dirs, bone_lens)

def bones2joints(bone_dirs, bone_lens, parents):
    ' Recover bone length and direction to joints location.\n\n        Parameters\n        ----------\n        bone_dirs: jt.tensor 1x24x3\n        bone_lens: jt.tensor Bx24x1\n    '
    batch_size = bone_lens.shape[0]
    joints = jt.zeros_like(bone_dirs).expand(batch_size, 24, 3)
    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if (p_id == (- 1)):
            joints[:, c_id] = bone_dirs[:, c_id]
        else:
            joints[:, c_id] = (joints[:, p_id] + (bone_dirs[:, c_id] * bone_lens[:, c_id]))
    return joints

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, lbs_weights, pose2rot=True, dtype=jt.float32):
    ' Performs Linear Blend Skinning with the given shape and pose parameters\n\n        Parameters\n        ----------\n        betas : jt.tensor BxNB\n            The tensor of shape parameters\n        pose : jt.tensor Bx(J + 1) * 3\n            The pose parameters in axis-angle format\n        v_template jt.tensor BxVx3\n            The template mesh that will be deformed\n        shapedirs : jt.tensor 1xNB\n            The tensor of PCA shape displacements\n        posedirs : jt.tensor Px(V * 3)\n            The pose PCA coefficients\n        J_regressor : jt.tensor JxV\n            The regressor array that is used to calculate the joints from\n            the position of the vertices\n        parents: jt.tensor J\n            The array that describes the kinematic tree for the model\n        lbs_weights: jt.tensor N x V x (J + 1)\n            The linear blend skinning weights that represent how much the\n            rotation matrix of each part affects each vertex\n        pose2rot: bool, optional\n            Flag on whether to convert the input pose tensor to rotation\n            matrices. The default value is True. If False, then the pose tensor\n            should already contain rotation matrices and have a size of\n            Bx(J + 1)x9\n        dtype: jt.dtype, optional\n\n        Returns\n        -------\n        verts: jt.tensor BxVx3\n            The vertices of the mesh after applying the shape and pose\n            displacements.\n        joints: jt.tensor BxJx3\n            The joints of the model\n        rot_mats: jt.tensor BxJx3x3\n            The rotation matrics of each joints\n    '
    batch_size = max(betas.shape[0], pose.shape[0])
    # device = betas.device
    v_shaped = (v_template + blend_shapes(betas, shapedirs))
    J = vertices2joints(J_regressor, v_shaped)
    ident = jt.eye(3, dtype=dtype)
    if pose2rot:
        if (pose.numel() == ((batch_size * 24) * 4)):
            rot_mats = quat_to_rotmat(pose.reshape((batch_size * 24), 4)).reshape((batch_size, 24, 3, 3))
        else:
            rot_mats = batch_rodrigues(pose.view((- 1), 3), dtype=dtype).view([batch_size, (- 1), 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, (- 1)])
        pose_offsets = jt.matmul(pose_feature, posedirs).view((batch_size, (- 1), 3))
    else:
        pose_feature = (pose[:, 1:].view((batch_size, (- 1), 3, 3)) - ident)
        rot_mats = pose.view((batch_size, (- 1), 3, 3))
        pose_offsets = jt.matmul(pose_feature.view(batch_size, (- 1)), posedirs).view((batch_size, (- 1), 3))
    v_posed = (pose_offsets + v_shaped)
    (J_transformed, A) = batch_rigid_transform(rot_mats, J, parents[:24], dtype=dtype)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, (- 1), (- 1)])
    num_joints = J_regressor.shape[0]
    T = jt.matmul(W, A.view(batch_size, num_joints, 16)).view((batch_size, (- 1), 4, 4))
    homogen_coord = jt.ones([batch_size, v_posed.shape[1], 1], dtype=dtype)
    v_posed_homo = jt.contrib.concat([v_posed, homogen_coord], dim=2)
    v_homo = jt.matmul(T, jt.unsqueeze(v_posed_homo, dim=(- 1)))
    verts = v_homo[:, :, :3, 0]
    J_from_verts = vertices2joints(J_regressor_h36m, verts)
    return (verts, J_transformed, rot_mats, J_from_verts)

def hybrik(betas, global_orient, pose_skeleton, phis, v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, children, lbs_weights, dtype=jt.float32, train=False, leaf_thetas=None):
    ' Performs Linear Blend Skinning with the given shape and skeleton joints\n\n        Parameters\n        ----------\n        betas : jt.tensor BxNB\n            The tensor of shape parameters\n        global_orient : jt.tensor Bx3\n            The tensor of global orientation\n        pose_skeleton : jt.tensor BxJ*3\n            The pose skeleton in (X, Y, Z) format\n        phis : jt.tensor BxJx2\n            The rotation on bone axis parameters\n        v_template jt.tensor BxVx3\n            The template mesh that will be deformed\n        shapedirs : jt.tensor 1xNB\n            The tensor of PCA shape displacements\n        posedirs : jt.tensor Px(V * 3)\n            The pose PCA coefficients\n        J_regressor : jt.tensor JxV\n            The regressor array that is used to calculate the joints from\n            the position of the vertices\n        J_regressor_h36m : jt.tensor 17xV\n            The regressor array that is used to calculate the 17 Human3.6M joints from\n            the position of the vertices\n        parents: jt.tensor J\n            The array that describes the kinematic parents for the model\n        children: dict\n            The dictionary that describes the kinematic chidrens for the model\n        lbs_weights: jt.tensor N x V x (J + 1)\n            The linear blend skinning weights that represent how much the\n            rotation matrix of each part affects each vertex\n        dtype: jt.dtype, optional\n\n        Returns\n        -------\n        verts: jt.tensor BxVx3\n            The vertices of the mesh after applying the shape and pose\n            displacements.\n        joints: jt.tensor BxJx3\n            The joints of the model\n        rot_mats: jt.tensor BxJx3x3\n            The rotation matrics of each joints\n    '
    batch_size = max(betas.shape[0], pose_skeleton.shape[0])
    # device = betas.device
    v_shaped = (v_template + blend_shapes(betas, shapedirs))
    if (leaf_thetas is not None):
        rest_J = vertices2joints(J_regressor, v_shaped)
    else:
        rest_J = jt.zeros((v_shaped.shape[0], 29, 3), dtype=dtype)
        rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)
        leaf_number = [411, 2445, 5905, 3216, 6617]
        leaf_vertices = v_shaped[:, leaf_number].clone()
        rest_J[:, 24:] = leaf_vertices
    if train:
        (rot_mats, rotate_rest_pose) = batch_inverse_kinematics_transform(pose_skeleton, global_orient, phis, rest_J.clone(), children, parents, dtype=dtype, train=train, leaf_thetas=leaf_thetas)
    else:
        (rot_mats, rotate_rest_pose) = batch_inverse_kinematics_transform_optimized(pose_skeleton, phis, rest_J.clone(), children, parents, dtype=dtype, train=train, leaf_thetas=leaf_thetas)
    test_joints = True
    if test_joints:
        (J_transformed, A) = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), parents[:24], dtype=dtype)
    else:
        J_transformed = None
    ident = jt.eye(3, dtype=dtype)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, (- 1)])
    pose_offsets = jt.matmul(pose_feature, posedirs).view((batch_size, (- 1), 3))
    v_posed = (pose_offsets + v_shaped)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, (- 1), (- 1)])
    num_joints = J_regressor.shape[0]
    T = jt.matmul(W, A.view(batch_size, num_joints, 16)).view((batch_size, (- 1), 4, 4))
    homogen_coord = jt.ones([batch_size, v_posed.shape[1], 1], dtype=dtype)
    v_posed_homo = jt.contrib.concat([v_posed, homogen_coord], dim=2)
    v_homo = jt.matmul(T, jt.unsqueeze(v_posed_homo, dim=(- 1)))
    verts = v_homo[:, :, :3, 0]
    J_from_verts_h36m = vertices2joints(J_regressor_h36m, verts)
    return (verts, J_transformed, rot_mats, J_from_verts_h36m)

def vertices2joints(J_regressor, vertices):
    ' Calculates the 3D joint locations from the vertices\n\n    Parameters\n    ----------\n    J_regressor : jt.tensor JxV\n        The regressor array that is used to calculate the joints from the\n        position of the vertices\n    vertices : jt.tensor BxVx3\n        The tensor of mesh vertices\n\n    Returns\n    -------\n    jt.tensor BxJx3\n        The location of the joints\n    '
    return jt.einsum('bik,ji->bjk', [vertices, J_regressor])

def blend_shapes(betas, shape_disps):
    ' Calculates the per vertex displacement due to the blend shapes\n\n\n    Parameters\n    ----------\n    betas : jt.tensor Bx(num_betas)\n        Blend shape coefficients\n    shape_disps: jt.tensor Vx3x(num_betas)\n        Blend shapes\n\n    Returns\n    -------\n    jt.tensor BxVx3\n        The per-vertex displacement due to shape deformation\n    '
    blend_shape = jt.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape

def batch_rodrigues(rot_vecs, epsilon=1e-08, dtype=jt.float32):
    ' Calculates the rotation matrices for a batch of rotation vectors\n        Parameters\n        ----------\n        rot_vecs: jt.tensor Nx3\n            array of N axis-angle vectors\n        Returns\n        -------\n        R: jt.tensor Nx3x3\n            The rotation matrices for the given axis-angle parameters\n    '
    batch_size = rot_vecs.shape[0]
    # device = rot_vecs.device
    angle = jt.norm((rot_vecs + 1e-08), dim=1, keepdims=True)
    rot_dir = (rot_vecs / angle)
    cos = jt.unsqueeze(jt.cos(angle), dim=1)
    sin = jt.unsqueeze(jt.sin(angle), dim=1)
    (rx, ry, rz) = jt.split(rot_dir, 1, dim=1)
    K = jt.zeros((batch_size, 3, 3), dtype=dtype)
    zeros = jt.zeros((batch_size, 1), dtype=dtype)
    K = jt.cat([zeros, (- rz), ry, rz, zeros, (- rx), (- ry), rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = jt.eye(3, dtype=dtype).unsqueeze(dim=0)
    rot_mat = ((ident + (sin * K)) + ((1 - cos) * jt.bmm(K, K)))
    return rot_mat

def transform_mat(R, t):
    ' Creates a batch of transformation matrices\n        Args:\n            - R: Bx3x3 array of a batch of rotation matrices\n            - t: Bx3x1 array of a batch of translation vectors\n        Returns:\n            - T: Bx4x4 Transformation matrix\n    '
    return jt.contrib.concat([None.pad(R, [0, 0, 0, 1]), None.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=jt.float32):
    '\n    Applies a batch of rigid transformations to the joints\n\n    Parameters\n    ----------\n    rot_mats : jt.tensor BxNx3x3\n        Tensor of rotation matrices\n    joints : jt.tensor BxNx3\n        Locations of joints. (Template Pose)\n    parents : jt.tensor BxN\n        The kinematic tree of each object\n    dtype : jt.dtype, optional:\n        The data type of the created tensors, the default is jt.float32\n\n    Returns\n    -------\n    posed_joints : jt.tensor BxNx3\n        The locations of the joints after applying the pose rotations\n    rel_transforms : jt.tensor BxNx4x4\n        The relative (with respect to the root joint) rigid transformations\n        for all the joints\n    '
    joints = jt.unsqueeze(joints, dim=(- 1))
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()
    transforms_mat = transform_mat(rot_mats.reshape((- 1), 3, 3), rel_joints.reshape((- 1), 3, 1)).reshape(((- 1), joints.shape[1], 4, 4))
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = jt.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = jt.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = None.pad(joints, [0, 0, 0, 1])
    rel_transforms = (transforms - None.pad(jt.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]))
    return (posed_joints, rel_transforms)

def batch_inverse_kinematics_transform(pose_skeleton, global_orient, phis, rest_pose, children, parents, dtype=jt.float32, train=False, leaf_thetas=None):
    '\n    Applies a batch of inverse kinematics transfoirm to the joints\n\n    Parameters\n    ----------\n    pose_skeleton : jt.tensor BxNx3\n        Locations of estimated pose skeleton.\n    global_orient : jt.tensor Bx1x3x3\n        Tensor of global rotation matrices\n    phis : jt.tensor BxNx2\n        The rotation on bone axis parameters\n    rest_pose : jt.tensor Bx(N+1)x3\n        Locations of rest_pose. (Template Pose)\n    children: dict\n        The dictionary that describes the kinematic chidrens for the model\n    parents : jt.tensor Bx(N+1)\n        The kinematic tree of each object\n    dtype : jt.dtype, optional:\n        The data type of the created tensors, the default is jt.float32\n\n    Returns\n    -------\n    rot_mats: jt.tensor Bx(N+1)x3x3\n        The rotation matrics of each joints\n    rel_transforms : jt.tensor Bx(N+1)x4x4\n        The relative (with respect to the root joint) rigid transformations\n        for all the joints\n    '
    batch_size = pose_skeleton.shape[0]
    # device = pose_skeleton.device
    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = jt.unsqueeze(rel_rest_pose, dim=(- 1))
    rotate_rest_pose = jt.zeros_like(rel_rest_pose)
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]
    rel_pose_skeleton = jt.unsqueeze(pose_skeleton.clone(), dim=(- 1)).detach()
    rel_pose_skeleton[:, 1:] = (rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone())
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]
    final_pose_skeleton = jt.unsqueeze(pose_skeleton.clone(), dim=(- 1))
    final_pose_skeleton = ((final_pose_skeleton - final_pose_skeleton[:, 0:1]) + rel_rest_pose[:, 0:1])
    rel_rest_pose = rel_rest_pose
    rel_pose_skeleton = rel_pose_skeleton
    final_pose_skeleton = final_pose_skeleton
    rotate_rest_pose = rotate_rest_pose
    assert (phis.ndim == 3)
    phis = (phis / (jt.norm(phis, dim=2, keepdims=True) + 1e-08))
    if train:
        global_orient_mat = batch_get_pelvis_orient(rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    else:
        global_orient_mat = batch_get_pelvis_orient_svd(rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    rot_mat_chain = [global_orient_mat]
    rot_mat_local = [global_orient_mat]
    if (leaf_thetas is not None):
        leaf_cnt = 0
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])
    for i in range(1, parents.shape[0]):
        if (children[i] == (- 1)):
            if (leaf_thetas is not None):
                rot_mat = leaf_rot_mats[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rotate_rest_pose[:, i] = (rotate_rest_pose[:, parents[i]] + jt.matmul(rot_mat_chain[parents[i]], rel_rest_pose[:, i]))
                rot_mat_chain.append(jt.matmul(rot_mat_chain[parents[i]], rot_mat))
                rot_mat_local.append(rot_mat)
        elif (children[i] == (- 3)):
            rotate_rest_pose[:, i] = (rotate_rest_pose[:, parents[i]] + jt.matmul(rot_mat_chain[parents[i]], rel_rest_pose[:, i]))
            spine_child = []
            for c in range(1, parents.shape[0]):
                if ((parents[c] == i) and (c not in spine_child)):
                    spine_child.append(c)
            spine_child = []
            for c in range(1, parents.shape[0]):
                if ((parents[c] == i) and (c not in spine_child)):
                    spine_child.append(c)
            children_final_loc = []
            children_rest_loc = []
            for c in spine_child:
                temp = (final_pose_skeleton[:, c] - rotate_rest_pose[:, i])
                children_final_loc.append(temp)
                children_rest_loc.append(rel_rest_pose[:, c].clone())
            rot_mat = batch_get_3children_orient_svd(children_final_loc, children_rest_loc, rot_mat_chain[parents[i]], spine_child, dtype)
            rot_mat_chain.append(jt.matmul(rot_mat_chain[parents[i]], rot_mat))
            rot_mat_local.append(rot_mat)
        else:
            rotate_rest_pose[:, i] = (rotate_rest_pose[:, parents[i]] + jt.matmul(rot_mat_chain[parents[i]], rel_rest_pose[:, i]))
            child_final_loc = (final_pose_skeleton[:, children[i]] - rotate_rest_pose[:, i])
            if (not train):
                orig_vec = rel_pose_skeleton[:, children[i]]
                template_vec = rel_rest_pose[:, children[i]]
                norm_t = jt.norm(template_vec, dim=1, keepdims=True)
                orig_vec = ((orig_vec * norm_t) / jt.norm(orig_vec, dim=1, keepdims=True))
                diff = jt.norm((child_final_loc - orig_vec), dim=1, keepdims=True)
                big_diff_idx = jt.where((diff > (15 / 1000)))[0]
                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
            child_final_loc = jt.matmul(rot_mat_chain[parents[i]].transpose(1, 2), child_final_loc)
            child_rest_loc = rel_rest_pose[:, children[i]]
            child_final_norm = jt.norm(child_final_loc, dim=1, keepdims=True)
            child_rest_norm = jt.norm(child_rest_loc, dim=1, keepdims=True)
            child_final_norm = jt.norm(child_final_loc, dim=1, keepdims=True)
            axis = jt.cross(child_rest_loc, child_final_loc, dim=1)
            axis_norm = jt.norm(axis, dim=1, keepdims=True)
            cos = (jt.sum((child_rest_loc * child_final_loc), dim=1, keepdims=True) / ((child_rest_norm * child_final_norm) + 1e-08))
            sin = (axis_norm / ((child_rest_norm * child_final_norm) + 1e-08))
            axis = (axis / (axis_norm + 1e-08))
            (rx, ry, rz) = jt.split(axis, 1, dim=1)
            zeros = jt.zeros((batch_size, 1, 1), dtype=dtype)
            K = jt.cat([zeros, (- rz), ry, rz, zeros, (- rx), (- ry), rx, zeros], dim=1).view((batch_size, 3, 3))
            ident = jt.eye(3, dtype=dtype).unsqueeze(dim=0)
            rot_mat_loc = ((ident + (sin * K)) + ((1 - cos) * jt.bmm(K, K)))
            spin_axis = (child_rest_loc / child_rest_norm)
            (rx, ry, rz) = jt.split(spin_axis, 1, dim=1)
            zeros = jt.zeros((batch_size, 1, 1), dtype=dtype)
            K = jt.cat([zeros, (- rz), ry, rz, zeros, (- rx), (- ry), rx, zeros], dim=1).view((batch_size, 3, 3))
            ident = jt.eye(3, dtype=dtype).unsqueeze(dim=0)
            (cos, sin) = jt.split(phis[:, (i - 1)], 1, dim=1)
            cos = jt.unsqueeze(cos, dim=2)
            sin = jt.unsqueeze(sin, dim=2)
            rot_mat_spin = ((ident + (sin * K)) + ((1 - cos) * jt.bmm(K, K)))
            rot_mat = jt.matmul(rot_mat_loc, rot_mat_spin)
            rot_mat_chain.append(jt.matmul(rot_mat_chain[parents[i]], rot_mat))
            rot_mat_local.append(rot_mat)
    rot_mats = jt.stack(rot_mat_local, dim=1)
    return (rot_mats, rotate_rest_pose.squeeze((- 1)))

def batch_inverse_kinematics_transform_optimized(pose_skeleton, phis, rest_pose, children, parents, dtype=jt.float32, train=False, leaf_thetas=None):
    '\n    Applies a batch of inverse kinematics transfoirm to the joints\n\n    Parameters\n    ----------\n    pose_skeleton : jt.tensor BxNx3\n        Locations of estimated pose skeleton.\n    global_orient : jt.tensor Bx1x3x3\n        Tensor of global rotation matrices\n    phis : jt.tensor BxNx2\n        The rotation on bone axis parameters\n    rest_pose : jt.tensor Bx(N+1)x3\n        Locations of rest_pose. (Template Pose)\n    children: dict\n        The dictionary that describes the kinematic chidrens for the model\n    parents : jt.tensor Bx(N+1)\n        The kinematic tree of each object\n    dtype : jt.dtype, optional:\n        The data type of the created tensors, the default is jt.float32\n\n    Returns\n    -------\n    rot_mats: jt.tensor Bx(N+1)x3x3\n        The rotation matrics of each joints\n    rel_transforms : jt.tensor Bx(N+1)x4x4\n        The relative (with respect to the root joint) rigid transformations\n        for all the joints\n    '
    batch_size = pose_skeleton.shape[0]
    # device = pose_skeleton.device
    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = jt.unsqueeze(rel_rest_pose, dim=(- 1))
    rotate_rest_pose = jt.zeros_like(rel_rest_pose)
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]
    rel_pose_skeleton = jt.unsqueeze(pose_skeleton.clone(), dim=(- 1)).detach()
    rel_pose_skeleton[:, 1:] = (rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone())
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]
    final_pose_skeleton = jt.unsqueeze(pose_skeleton.clone(), dim=(- 1))
    final_pose_skeleton = ((final_pose_skeleton - final_pose_skeleton[:, [0]]) + rel_rest_pose[:, [0]])
    phis = (phis / (jt.norm(phis, dim=2, keepdims=True) + 1e-08))
    if train:
        global_orient_mat = batch_get_pelvis_orient(rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    else:
        global_orient_mat = batch_get_pelvis_orient_svd(rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    rot_mat_chain = jt.zeros((batch_size, 24, 3, 3), dtype=jt.float32)
    rot_mat_local = jt.zeros_like(rot_mat_chain)
    rot_mat_chain[:, 0] = global_orient_mat
    rot_mat_local[:, 0] = global_orient_mat
    if (leaf_thetas is not None):
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])
    idx_levs = [[0], [3], [6], [9], [1, 2, 12, 13, 14], [4, 5, 15, 16, 17], [7, 8, 18, 19], [10, 11, 20, 21], [22, 23], [24, 25, 26, 27, 28]]
    if (leaf_thetas is not None):
        idx_levs = idx_levs[:(- 1)]
    for idx_lev in range(1, len(idx_levs)):
        indices = idx_levs[idx_lev]
        if (idx_lev == (len(idx_levs) - 1)):
            if (leaf_thetas is not None):
                rot_mat = leaf_rot_mats[:, :, :, :]
                parent_indices = [15, 22, 23, 10, 11]
                rot_mat_local[:, parent_indices] = rot_mat
                if (jt.det(rot_mat) < 0).any():
                    print('Something wrong.')
        elif (idx_lev == 3):
            idx = indices[0]
            rotate_rest_pose[:, idx] = (rotate_rest_pose[:, parents[idx]] + jt.matmul(rot_mat_chain[:, parents[idx]], rel_rest_pose[:, idx]))
            spine_child = [12, 13, 14]
            children_final_loc = []
            children_rest_loc = []
            for c in spine_child:
                temp = (final_pose_skeleton[:, c] - rotate_rest_pose[:, idx])
                children_final_loc.append(temp)
                children_rest_loc.append(rel_rest_pose[:, c].clone())
            rot_mat = batch_get_3children_orient_svd(children_final_loc, children_rest_loc, rot_mat_chain[:, parents[idx]], spine_child, dtype)
            rot_mat_chain[:, idx] = jt.matmul(rot_mat_chain[:, parents[idx]], rot_mat)
            rot_mat_local[:, idx] = rot_mat
            if (jt.det(rot_mat) < 0).any():
                print(1)
        else:
            len_indices = len(indices)
            rotate_rest_pose[:, indices] = (rotate_rest_pose[:, parents[indices]] + jt.matmul(rot_mat_chain[:, parents[indices]], rel_rest_pose[:, indices]))
            child_final_loc = (final_pose_skeleton[:, children[indices]] - rotate_rest_pose[:, indices])
            if (not train):
                orig_vec = rel_pose_skeleton[:, children[indices]]
                template_vec = rel_rest_pose[:, children[indices]]
                norm_t = jt.norm(template_vec, dim=2, keepdims=True)
                orig_vec = ((orig_vec * norm_t) / jt.norm(orig_vec, dim=2, keepdims=True))
                diff = jt.norm((child_final_loc - orig_vec), dim=2, keepdims=True).reshape((- 1))
                big_diff_idx = jt.where((diff > (15 / 1000)))[0]
                child_final_loc = child_final_loc.reshape(((batch_size * len_indices), 3, 1))
                orig_vec = orig_vec.reshape(((batch_size * len_indices), 3, 1))
                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape((batch_size, len_indices, 3, 1))
            child_final_loc = jt.matmul(rot_mat_chain[:, parents[indices]].transpose(2, 3), child_final_loc)
            child_rest_loc = rel_rest_pose[:, children[indices]]
            child_final_norm = jt.norm(child_final_loc, dim=2, keepdims=True)
            child_rest_norm = jt.norm(child_rest_loc, dim=2, keepdims=True)
            axis = jt.cross(child_rest_loc, child_final_loc, dim=2)
            axis_norm = jt.norm(axis, dim=2, keepdims=True)
            cos = (jt.sum((child_rest_loc * child_final_loc), dim=2, keepdims=True) / ((child_rest_norm * child_final_norm) + 1e-08))
            sin = (axis_norm / ((child_rest_norm * child_final_norm) + 1e-08))
            axis = (axis / (axis_norm + 1e-08))
            (rx, ry, rz) = jt.split(axis, 1, dim=2)
            zeros = jt.zeros((batch_size, len_indices, 1, 1), dtype=dtype)
            K = jt.cat([zeros, (- rz), ry, rz, zeros, (- rx), (- ry), rx, zeros], dim=2).view((batch_size, len_indices, 3, 3))
            ident = jt.eye(3, dtype=dtype).reshape((1, 1, 3, 3))
            rot_mat_loc = ((ident + (sin * K)) + ((1 - cos) * jt.matmul(K, K)))
            spin_axis = (child_rest_loc / child_rest_norm)
            (rx, ry, rz) = jt.split(spin_axis, 1, dim=2)
            zeros = jt.zeros((batch_size, len_indices, 1, 1), dtype=dtype)
            K = jt.cat([zeros, (- rz), ry, rz, zeros, (- rx), (- ry), rx, zeros], dim=2).view((batch_size, len_indices, 3, 3))
            ident = jt.eye(3, dtype=dtype).reshape((1, 1, 3, 3))
            phi_indices = [(item - 1) for item in indices]
            (cos, sin) = jt.split(phis[:, phi_indices], 1, dim=2)
            cos = jt.unsqueeze(cos, dim=3)
            sin = jt.unsqueeze(sin, dim=3)
            rot_mat_spin = ((ident + (sin * K)) + ((1 - cos) * jt.matmul(K, K)))
            rot_mat = jt.matmul(rot_mat_loc, rot_mat_spin)
            if (jt.det(rot_mat) < 0).any():
                print(2, (jt.det(rot_mat_loc) < 0), (jt.det(rot_mat_spin) < 0))
            rot_mat_chain[:, indices] = jt.matmul(rot_mat_chain[:, parents[indices]], rot_mat)
            rot_mat_local[:, indices] = rot_mat
    rot_mats = rot_mat_local
    return (rot_mats, rotate_rest_pose.squeeze((- 1)))

def batch_get_pelvis_orient_svd(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if ((parents[i] == 0) and (i not in pelvis_child)):
            pelvis_child.append(i)
    rest_mat = []
    target_mat = []
    for child in pelvis_child:
        rest_mat.append(rel_rest_pose[:, child].clone())
        target_mat.append(rel_pose_skeleton[:, child].clone())
    rest_mat = jt.contrib.concat(rest_mat, dim=2)
    target_mat = jt.contrib.concat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))
    mask_zero = S.sum(dim=(1, 2))
    S_non_zero = S[(mask_zero != 0)].reshape(((- 1), 3, 3))
    (U, _, V) = jt.svd(S_non_zero)
    rot_mat = jt.zeros_like(S)
    rot_mat[(mask_zero == 0)] = jt.eye(3)
    det_u_v = jt.det(jt.bmm(V, U.transpose(1, 2)))
    det_modify_mat = jt.eye(3).unsqueeze(0).expand(U.shape[0], (- 1), (- 1)).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat_non_zero = jt.bmm(jt.bmm(V, det_modify_mat), U.transpose(1, 2))
    rot_mat[(mask_zero != 0)] = rot_mat_non_zero
    assert (jt.sum(jt.isnan(rot_mat)) == 0), ('rot_mat', rot_mat)
    return rot_mat

def batch_get_pelvis_orient(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    # device = rel_pose_skeleton.device
    assert (children[0] == 3)
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if ((parents[i] == 0) and (i not in pelvis_child)):
            pelvis_child.append(i)
    spine_final_loc = rel_pose_skeleton[:, int(children[0])].clone()
    spine_rest_loc = rel_rest_pose[:, int(children[0])].clone()
    spine_norm = jt.norm(spine_final_loc, dim=1, keepdims=True)
    spine_norm = (spine_final_loc / (spine_norm + 1e-08))
    rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype)
    assert (jt.sum(jt.isnan(rot_mat_spine)) == 0), ('rot_mat_spine', rot_mat_spine)
    center_final_loc = 0
    center_rest_loc = 0
    for child in pelvis_child:
        if (child == int(children[0])):
            continue
        center_final_loc = (center_final_loc + rel_pose_skeleton[:, child].clone())
        center_rest_loc = (center_rest_loc + rel_rest_pose[:, child].clone())
    center_final_loc = (center_final_loc / (len(pelvis_child) - 1))
    center_rest_loc = (center_rest_loc / (len(pelvis_child) - 1))
    center_rest_loc = jt.matmul(rot_mat_spine, center_rest_loc)
    center_final_loc = (center_final_loc - (jt.sum((center_final_loc * spine_norm), dim=1, keepdims=True) * spine_norm))
    center_rest_loc = (center_rest_loc - (jt.sum((center_rest_loc * spine_norm), dim=1, keepdims=True) * spine_norm))
    center_final_loc_norm = jt.norm(center_final_loc, dim=1, keepdims=True)
    center_rest_loc_norm = jt.norm(center_rest_loc, dim=1, keepdims=True)
    axis = jt.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = jt.norm(axis, dim=1, keepdims=True)
    cos = (jt.sum((center_rest_loc * center_final_loc), dim=1, keepdims=True) / ((center_rest_loc_norm * center_final_loc_norm) + 1e-08))
    sin = (axis_norm / ((center_rest_loc_norm * center_final_loc_norm) + 1e-08))
    assert (jt.sum(jt.isnan(cos)) == 0), ('cos', cos)
    assert (jt.sum(jt.isnan(sin)) == 0), ('sin', sin)
    axis = (axis / (axis_norm + 1e-08))
    (rx, ry, rz) = jt.split(axis, 1, dim=1)
    zeros = jt.zeros((batch_size, 1, 1), dtype=dtype)
    K = jt.cat([zeros, (- rz), ry, rz, zeros, (- rx), (- ry), rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = jt.eye(3, dtype=dtype).unsqueeze(dim=0)
    rot_mat_center = ((ident + (sin * K)) + ((1 - cos) * jt.bmm(K, K)))
    rot_mat = jt.matmul(rot_mat_center, rot_mat_spine)
    return rot_mat

def batch_get_3children_orient_svd(rel_pose_skeleton, rel_rest_pose, rot_mat_chain_parent, children_list, dtype):
    rest_mat = []
    target_mat = []
    for (c, child) in enumerate(children_list):
        if isinstance(rel_pose_skeleton, list):
            target = rel_pose_skeleton[c].clone()
            template = rel_rest_pose[c].clone()
        else:
            target = rel_pose_skeleton[:, child].clone()
            template = rel_rest_pose[:, child].clone()
        target = jt.matmul(rot_mat_chain_parent.transpose(1, 2), target)
        target_mat.append(target)
        rest_mat.append(template)
    rest_mat = jt.contrib.concat(rest_mat, dim=2)
    target_mat = jt.contrib.concat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))
    (U, _, V) = jt.svd(S)
    det_u_v = jt.det(jt.bmm(V, U.transpose(1, 2)))
    det_modify_mat = jt.eye(3).unsqueeze(0).expand(U.shape[0], (- 1), (- 1)).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat = jt.bmm(jt.bmm(V, det_modify_mat), U.transpose(1, 2))
    assert (jt.sum(jt.isnan(rot_mat)) == 0), ('3children rot_mat', rot_mat)
    return rot_mat

def vectors2rotmat(vec_rest, vec_final, dtype):
    batch_size = vec_final.shape[0]
    # device = vec_final.device
    vec_final_norm = jt.norm(vec_final, dim=1, keepdims=True)
    vec_rest_norm = jt.norm(vec_rest, dim=1, keepdims=True)
    axis = jt.cross(vec_rest, vec_final, dim=1)
    axis_norm = jt.norm(axis, dim=1, keepdims=True)
    cos = (jt.sum((vec_rest * vec_final), dim=1, keepdims=True) / ((vec_rest_norm * vec_final_norm) + 1e-08))
    sin = (axis_norm / ((vec_rest_norm * vec_final_norm) + 1e-08))
    axis = (axis / (axis_norm + 1e-08))
    (rx, ry, rz) = jt.split(axis, 1, dim=1)
    zeros = jt.zeros((batch_size, 1, 1), dtype=dtype)
    K = jt.cat([zeros, (- rz), ry, rz, zeros, (- rx), (- ry), rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = jt.eye(3, dtype=dtype).unsqueeze(dim=0)
    rot_mat_loc = ((ident + (sin * K)) + ((1 - cos) * jt.bmm(K, K)))
    return rot_mat_loc

def rotmat_to_quat(rotation_matrix):
    assert (rotation_matrix.shape[1:] == (3, 3))
    rot_mat = rotation_matrix.reshape(((- 1), 3, 3))
    hom = jt.array([0, 0, 1])
    hom = hom.reshape((1, 3, 1)).expand(rot_mat.shape[0], (- 1), (- 1))
    rotation_matrix = jt.contrib.concat([rot_mat, hom], dim=(- 1))
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-06):
    '\n    This function is borrowed from https://github.com/kornia/kornia\n\n    Convert 3x4 rotation matrix to 4d quaternion vector\n\n    This algorithm is based on algorithm described in\n    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201\n\n    Args:\n        rotation_matrix (Tensor): the rotation matrix to convert.\n\n    Return:\n        Tensor: the rotation in quaternion\n\n    Shape:\n        - Input: :math:`(N, 3, 4)`\n        - Output: :math:`(N, 4)`\n\n    Example:\n        >>> input = jt.rand(4, 3, 4)  # Nx3x4\n        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4\n    '
    if (not jt.is_tensor(rotation_matrix)):
        raise TypeError('Input type is not a jt.Tensor. Got {}'.format(type(rotation_matrix)))
    if (len(rotation_matrix.shape) > 3):
        raise ValueError('Input size must be a three dimensional tensor. Got {}'.format(rotation_matrix.shape))
    if (not (rotation_matrix.shape[(- 2):] == (3, 4))):
        raise ValueError('Input size must be a N x 3 x 4  tensor. Got {}'.format(rotation_matrix.shape))
    rmat_t = jt.transpose(rotation_matrix, 1, 2)
    mask_d2 = (rmat_t[:, 2, 2] < eps)
    mask_d0_d1 = (rmat_t[:, 0, 0] > rmat_t[:, 1, 1])
    mask_d0_nd1 = (rmat_t[:, 0, 0] < (- rmat_t[:, 1, 1]))
    t0 = (((1 + rmat_t[:, 0, 0]) - rmat_t[:, 1, 1]) - rmat_t[:, 2, 2])
    q0 = jt.stack([(rmat_t[:, 1, 2] - rmat_t[:, 2, 1]), t0, (rmat_t[:, 0, 1] + rmat_t[:, 1, 0]), (rmat_t[:, 2, 0] + rmat_t[:, 0, 2])], dim=(- 1))
    t0_rep = t0.repeat(4, 1).t()
    t1 = (((1 - rmat_t[:, 0, 0]) + rmat_t[:, 1, 1]) - rmat_t[:, 2, 2])
    q1 = jt.stack([(rmat_t[:, 2, 0] - rmat_t[:, 0, 2]), (rmat_t[:, 0, 1] + rmat_t[:, 1, 0]), t1, (rmat_t[:, 1, 2] + rmat_t[:, 2, 1])], dim=(- 1))
    t1_rep = t1.repeat(4, 1).t()
    t2 = (((1 - rmat_t[:, 0, 0]) - rmat_t[:, 1, 1]) + rmat_t[:, 2, 2])
    q2 = jt.stack([(rmat_t[:, 0, 1] - rmat_t[:, 1, 0]), (rmat_t[:, 2, 0] + rmat_t[:, 0, 2]), (rmat_t[:, 1, 2] + rmat_t[:, 2, 1]), t2], dim=(- 1))
    t2_rep = t2.repeat(4, 1).t()
    t3 = (((1 + rmat_t[:, 0, 0]) + rmat_t[:, 1, 1]) + rmat_t[:, 2, 2])
    q3 = jt.stack([t3, (rmat_t[:, 1, 2] - rmat_t[:, 2, 1]), (rmat_t[:, 2, 0] - rmat_t[:, 0, 2]), (rmat_t[:, 0, 1] - rmat_t[:, 1, 0])], dim=(- 1))
    t3_rep = t3.repeat(4, 1).t()
    mask_c0 = (mask_d2 * mask_d0_d1)
    mask_c1 = (mask_d2 * (~ mask_d0_d1))
    mask_c2 = ((~ mask_d2) * mask_d0_nd1)
    mask_c3 = ((~ mask_d2) * (~ mask_d0_nd1))
    mask_c0 = mask_c0.view(((- 1), 1)).type_as(q0)
    mask_c1 = mask_c1.view(((- 1), 1)).type_as(q1)
    mask_c2 = mask_c2.view(((- 1), 1)).type_as(q2)
    mask_c3 = mask_c3.view(((- 1), 1)).type_as(q3)
    q = ((((q0 * mask_c0) + (q1 * mask_c1)) + (q2 * mask_c2)) + (q3 * mask_c3))
    q /= jt.sqrt(((((t0_rep * mask_c0) + (t1_rep * mask_c1)) + (t2_rep * mask_c2)) + (t3_rep * mask_c3)))
    q *= 0.5
    return q

def quat_to_rotmat(quat):
    'Convert quaternion coefficients to rotation matrix.\n    Args:\n        quat: size = [B, 4] 4 <===>(w, x, y, z)\n    Returns:\n        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]\n    '
    norm_quat = quat
    norm_quat = (norm_quat / (norm_quat.norm(p=2, dim=1, keepdims=True) + 1e-08))
    (w, x, y, z) = (norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3])
    B = quat.shape[0]
    (w2, x2, y2, z2) = (w.pow(2), x.pow(2), y.pow(2), z.pow(2))
    (wx, wy, wz) = ((w * x), (w * y), (w * z))
    (xy, xz, yz) = ((x * y), (x * z), (y * z))
    rotMat = jt.stack([(((w2 + x2) - y2) - z2), ((2 * xy) - (2 * wz)), ((2 * wy) + (2 * xz)), ((2 * wz) + (2 * xy)), (((w2 - x2) + y2) - z2), ((2 * yz) - (2 * wx)), ((2 * xz) - (2 * wy)), ((2 * wx) + (2 * yz)), (((w2 - x2) - y2) + z2)], dim=1).view((B, 3, 3))
    return rotMat
