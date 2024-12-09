from pytorch3d.renderer import (
    PointLights, diffuse, specular
)
from pytorch3d.transforms.transform3d import Transform3d
import torch
from nnutils.geom_utils import bone_transform, gauss_mlp_skinning

def _validate_light_properties(obj) -> None:
    props = ("ambient_color", "diffuse_color", "specular_color")
    for n in props:
        t = getattr(obj, n)
        if t.shape[-1] != 3:
            msg = "Expected %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))

class PointLightsIntensity(PointLights):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        location=((0, 1, 0),),
        device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            location=location,
        )
        _validate_light_properties(self)
        if self.location.shape[-1] != 3:
            msg = "Expected location to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.location.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def reshape_location(self, points) -> torch.Tensor:
        """
        Reshape the location tensor to have dimensions
        compatible with the points which can either be of
        shape (P, 3) or (N, H, W, K, 3).
        """
        if self.location.ndim == points.ndim:
            # pyre-fixme[7]
            return self.location
        # pyre-fixme[29]
        return self.location[:, None, None, None, :]

    def relocate(self):
        pass

    def diffuse(self, normals, points) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=direction,
            camera_position=camera_position,
            shininess=shininess,
        )

def get_lbs_fw_mat(
        points,
        bones,
        bone_rts_fw,
        skin_aux,
        rest_pose_code,
        time_embedded,
        embedding_xyz,
        device = None
    ) -> None:
        """
        Create a PyTorch3d transform based on a linear blend skinning model.
        """
        device_ = bones.get_device() if device == None else device
        
        # Compute weights skinning weights
        bones_rst = bones # Bones in canonical space
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device)) # Embedding of size 128
        nerf_skin = None # In the finer step this is controlled by the elasticity term

        time_embedded = time_embedded #rays['time_embedded'][:,None] # ?
        # coords after deform
        # bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True) # NO NEED. Bones after applying a forward (canonical -> bones at time t) rigid transformation
        skinning_weights = gauss_mlp_skinning(points, embedding_xyz, bones_rst, time_embedded,  nerf_skin, skin_aux=skin_aux) # Use bones at rest

        # backward skinning -> Obtain NeRF samples in canonical space? Yes (also returning here bones_dfm for some reason when they have already been computed)
        """xyz_coarse_sampled, bones_dfm = lbs(bones_rst, 
                                                  bone_rts_fw, 
                                                  skin_backward,
                                                  xyz_coarse_sampled,
                                                  )"""
        
        # Forward transformation J for points in canonical position to frame t
        rts_fw = bone_rts_fw
        rts = rts_fw
        B = rts.shape[-3]
        pts, skinning_weights = points.permute(1,0,2).expand(B, -1, -1), skinning_weights.permute(2, 0, 1).expand(-1, -1, B)
        N = pts.shape[-2]
        pts = pts.view(-1,N,3)
        rts = rts.view(-1,B,3,4)
        Rmat = rts[:,:,:3,:3] # bs, B, 3,3
        Tmat = rts[:,:,:3,3]
        device = Tmat.device
        # skinning_weights (B, N, 3) # skin: bs,N,B   - skinning matrix

        Rmat, Tmat, skinning_weights = Rmat.permute(1,0,2,3), Tmat.permute(1, 0, 2), skinning_weights.permute(2, 1, 0)
        # Gi=sum(wbGb), V=RV+T
        Rmat_w = (skinning_weights[...,None,None] * Rmat[:,None]).sum(2) # bs,N,B,3
        Tmat_w = (skinning_weights[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3
        pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
        bones_dfm = bone_transform(bones, rts_fw) # bone coordinates after deform
        return pts, bones_dfm


class ElasticTransform(Transform3d):
    def __init__(
        self,
        points,
        device = None,
        orthogonal_tol: float = 1e-5,
    ) -> None:
        """
        Create a new Transform3d representing 3D rotation using a rotation
        matrix as the input.

        Args:
            R: a tensor of shape (3, 3) or (N, 3, 3)
            orthogonal_tol: tolerance for the test of the orthogonality of R

        """
        # TODO: Use somehow delta skinning weights (from MLP) ?
        device_ = points.get_device() if device == None else device
        super().__init__(device=device_)

    def _get_matrix_inverse(self) -> torch.Tensor:
        """
        Return the inverse of self._matrix.
        """
        return None