import tinycudann as tnn
import torch
from torch import nn
from copy import deepcopy

# Configuration for multiresolution hash encodings: https://github.com/nvlabs/instant-ngp
# log2_hashmap_size is an experiment-depending param (T parameter in paper https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)
hash_encoder_config = {
	"otype": "Grid",           # Component type.
	"type": "Hash",            # Type of backing storage of the
	                           # grids. Can be "Hash", "Tiled"
	                           # or "Dense".
	"n_levels": 16,            # Number of levels (resolutions)
	"n_features_per_level": 2, # Dimensionality of feature vector
	                           # stored in each level's entries.
	"log2_hashmap_size": 16,   # If type is "Hash", is the base-2
	                           # logarithm of the number of elements
	                           # in each backing hash table.
	"base_resolution": 16,     # The resolution of the coarsest le-
	                           # vel is base_resolution^input_dims.
    "per_level_scale": 2.0,    # The geometric growth factor, i.e.
	                           # the factor by which the resolution
	                           # of each grid is larger (per axis)
	                           # than that of the preceeding level.
	"interpolation": "Linear", # How to interpolate nearby grid
	                           # lookups. Can be "Nearest", "Linear",
	                           # or "Smoothstep" (for smooth deri-
	                           # vatives).
    "n_dims_to_encode": 3      # 3D coordinates                           
}

"""
	"per_level_scale": 2.0,    # The geometric growth factor, i.e.
	                           # the factor by which the resolution
	                           # of each grid is larger (per axis)
	                           # than that of the preceeding level.
	"interpolation": "Linear", # How to interpolate nearby grid
	                           # lookups. Can be "Nearest", "Linear",
	                           # or "Smoothstep" (for smooth deri-
	                           # vatives).
    "n_dims_to_encode": 3      # 3D coordinates
"""

oneblob_encoder_config = {
    "otype": "OneBlob",
    "n_dims_to_encode": 3,
    "n_bins": 4
}

dir_freq_encoder_config = {   
    "n_dims_to_encode": 3, # Viewing direction
    "otype": "Frequency", # Component type.
    "n_frequencies": 12   # Number of frequencies (sin & cos)
                            # per encoded dimension.
} # Was set to 12 by default, banmo is using 4

time_freq_encoder_config = {   
    "n_dims_to_encode": 1, # Viewing direction
    "otype": "Frequency", # Component type.
    "n_frequencies": 12   # Number of frequencies (sin & cos)
                            # per encoded dimension.
} # Was set to 12 by default, banmo is using 4

nerf_multihash_plus_dir_config = {
    "otype": "Composite",
    "nested": [
        {
            "n_dims_to_encode": 3, # Position
            **hash_encoder_config
        },
        {   
            "n_dims_to_encode": 3, # Viewing direction
            "otype": "Frequency", # Component type.
            "n_frequencies": 12   # Number of frequencies (sin & cos)
                                    # per encoded dimension.
        }
        #{
        #    "n_dims_to_encode": 3, # Interesting conditionals
        #    "otype": "OneBlob",    # One-blob encoding for direction. See https://arxiv.org/pdf/1808.03856.pdf, section 4.3
        #    "n_bins": 4            # In the original paper of one-blob encoding this is set to 32 
        #}
    ]
}

"""
One-blob encoding: An important consideration is the encoding of the inputs to the
network. We propose to use the one-blob encoding—a generalization
of the one-hot encoding [Harris and Harris 2013]—where a kernel
is used to activate multiple adjacent entries instead of a single one.
Assume a scalar s ∈ [0, 1] and a quantization of the unit interval into
k bins (we use k = 32). The one-blob encoding amounts to placing
a kernel (we use a Gaussian with σ = 1/k) at s and discretizing it
into the bins. With the proposed architecture of the neural network
(placement of ReLUs in particular, see Figure 3), the one-blob en-
coding effectively shuts down certain parts of the linear path of the
network, allowing it to specialize the model on various sub-domains
of the input.
"""

# n_layers / n_neurons are experiment-depending params
nerf_mlp_config = {
	"otype": "FullyFusedMLP",    # Component type. -> Fast MLP
	"activation": "ReLU",        # Activation of hidden layers.
	"output_activation": "None", # Activation of the output layer.
	"n_neurons": 128,            # Neurons in each hidden layer.
	                             # May only be 16, 32, 64, or 128.
	"n_hidden_layers": 5,        # Number of hidden layers.
	"feedback_alignment": False  # Use feedback alignment
	                             # [Lillicrap et al. 2016].
}

class TNNEncoder(nn.Module):
    def __init__(self, n_dims, opts):
        super(TNNEncoder, self).__init__()
        self.encoder = tnn.Encoding(n_dims, opts)
        self.alpha = 8

    def forward(self, x):
        ndim = x.ndim
        if ndim == 3: # Add case with environment code
            N_r, N_s, C = x.size()
        else:
            N_r = 1
            N_s, C = x.size()
            
        x = x.resize(N_r * N_s, C)
        out = self.encoder(x)

        if ndim == 3:
            out = out.resize(N_r, N_s, self.encoder.n_output_dims)

        return out.float()

class NeRFMultiResHashEncoder(nn.Module):
    def __init__(self, opts, n_input_dims=6+64, n_output_dims=4, init_beta=1./100):
        # n_input_dims = 6 (3D position + dir) + 64 (environment code)
        # n_output_dims = 4 (RGB + sigma)
        # Get config for dict mlp / encoder
        super(NeRFMultiResHashEncoder, self).__init__()
        self.hash_encoder_config = deepcopy(hash_encoder_config)
        self.dir_freq_encoder_config = deepcopy(dir_freq_encoder_config)
        self.mlp_config = deepcopy(nerf_mlp_config)

        self.hash_encoder_config["log2_hashmap_size"] = opts.log2_hashmap_size
        self.mlp_config["n_neurons"] = min(128, opts.nerf_hidden_dim)
        self.mlp_config["n_hidden_layers"] = opts.nerf_n_layers

        #self.nerf = tnn.NetworkWithInputEncoding(
        #                n_input_dims, n_output_dims,
        #                self.encoder_config, self.mlp_config
        #            )
        self.n_dim_env_code = n_input_dims-6
        self.spatial_encoding = TNNEncoder(3, self.hash_encoder_config)
        self.dir_encoding = TNNEncoder(3, self.dir_freq_encoder_config)
        self.dim_out_spat, self.dim_out_dir = self.spatial_encoding.encoder.n_output_dims, self.dir_encoding.encoder.n_output_dims
        self.mlp_network = tnn.Network(self.dim_out_spat + self.dim_out_dir + self.n_dim_env_code, \
                                        n_output_dims, self.mlp_config)

        self.beta = torch.Tensor([init_beta]) # logbeta
        self.beta = nn.Parameter(self.beta)

    def forward(self, x ,xyz=None, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_only:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        # Add 64-dim environment code
        # If viewing direction is not specified, add random direction -> encoded in range [-1, 1]. View dir not specified when initializing SDF to sphere
        ndim = x.ndim
        if ndim == 3: # Add case with environment code
            if x.size(2) == self.dim_out_spat:
                coords, viewing_dir, env_code = x,  (2 * (torch.rand(x.size(0), x.size(1), 3) - 0.5)).to(x.get_device()), (2 * (torch.rand(x.size(0), x.size(1), 64) - 0.5)).to(x.get_device())
                viewing_dir = self.dir_encoding(viewing_dir)
            elif x.size(2) == self.dim_out_spat + self.dim_out_dir:
                coords, viewing_dir, env_code = x[:, :, :self.dim_out_spat], x[:, :, self.dim_out_spat:], \
                    (2 * (torch.rand(x.size(0), x.size(1), 64) - 0.5)).to(x.get_device())
            else:
                coords, viewing_dir, env_code = x[:, :, :self.dim_out_spat], x[:, :, self.dim_out_spat:self.dim_out_spat+self.dim_out_dir], \
                                                x[:, :, self.dim_out_spat+self.dim_out_dir:]
            N_r, N_s, _ = coords.size()
        else:
            if x.size(1) == self.dim_out_spat:
                coords, viewing_dir, env_code = x, (2 * (torch.rand(x.size(0), 3) - 0.5)).to(x.get_device()), (2 * (torch.rand(x.size(0), 64) - 0.5)).to(x.get_device())
                viewing_dir = self.dir_encoding(viewing_dir)
            elif x.size(1) == self.dim_out_spat + self.dim_out_dir:
                coords, viewing_dir, env_code = x[:, :self.dim_out_spat], x[:, self.dim_out_spat:], (2 * (torch.rand(x.size(0), 64) - 0.5)).to(x.get_device())
            else:
                coords, viewing_dir, env_code = x[:, :self.dim_out_spat], x[:, self.dim_out_spat:self.dim_out_spat+self.dim_out_dir], \
                                                x[:, self.dim_out_spat+self.dim_out_dir:]
            N_r = 1
            N_s, _ = coords.size()
            
        out_spatial, out_vdir, env_code = coords.resize(N_r * N_s, self.dim_out_spat), viewing_dir.resize(N_r * N_s, self.dim_out_dir), env_code.resize(N_r * N_s, self.n_dim_env_code)
        encoding = torch.cat([out_spatial, out_vdir, env_code], 1)
        out = self.mlp_network(encoding).float()

        if ndim == 3:
            out = out.resize(N_r, N_s, 4)

            if sigma_only:
                return out[:, :, -1]
        
        if sigma_only:
            return out[:, -1]
        
        return out
