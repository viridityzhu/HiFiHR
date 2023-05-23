import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesAtlas, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    FoVPerspectiveCameras
)
import torchvision.transforms as transforms
from PIL import Image
# textured_pkl = "./assets/NIMBLE_TEX_FUV.pkl"
# textured_pkl = "utils/NIMBLE_model/assets/NIMBLE_TEX_FUV.pkl"
# f_uv = np.load(textured_pkl, allow_pickle=True)
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
root = '/home/jiayin/HandRecon/utils/NIMBLE_model/my_test/'
# Load the 3D model
verts, faces, aux = load_obj(root+"demo_0006_0000000_skin.obj",
                             create_texture_atlas=True)
verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
torch.save(verts_uvs, root+'verts_uvs.pt')
torch.save(faces_uvs, root+'faces_uvs.pt')
tex_maps = aux.texture_images
# tex_maps is a dictionary of {material name: texture image}.
# Take the first image:
texture_image = list(tex_maps.values())[0]
texture_image = texture_image[None, ...]  # (1, H, W, 3)
# Create a textures object
tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
# Create a Meshes object
mesh = Meshes( verts=[verts], faces=[faces.verts_idx], textures=tex)

# # load diffuse, specular and normal images
# diffuse_img = Image.open(root + "demo_0006_0000000_diffuse.png").convert("RGB")
# specular_img = Image.open(root + "demo_0006_0000000_spec.png").convert("RGB")
# normal_img = Image.open(root + "demo_0006_0000000_normal.png").convert("RGB")

# # convert images to PyTorch tensors
# diffuse_tensor = transforms.ToTensor()(diffuse_img).permute(2, 0, 1).unsqueeze(0)
# specular_tensor = transforms.ToTensor()(specular_img).permute(2, 0, 1).unsqueeze(0)
# normal_tensor = transforms.ToTensor()(normal_img).permute(2, 0, 1).unsqueeze(0)

# Generate a UV map
# uv = generate_textures_uvs([faces], [verts])

# textures = TexturesUV(maps=aux.texture_images, faces_uvs=[], verts_uvs=[aux.verts_uvs])

# Create the renderer
# renderer = SoftPhongRenderer(
#     device="cuda",
#     cameras=OpenGLPerspectiveCameras(device="cuda"),
# )
sigma = 1e-4
raster_settings_soft = RasterizationSettings(
    image_size=224, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    # perspective_correct=False, 
)

# # Differentiable soft renderer using per vertex RGB colors for texture
# renderer_textured = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=camera, 
#         raster_settings=raster_settings_soft
#     ),
#     shader=SoftPhongShader(device=device, 
#         cameras=camera,
#         lights=lights)
# create a renderer object
R, T = look_at_view_transform(dist=2.7, elev=10, azim=-15) 
cameras = FoVPerspectiveCameras( R=R, T=T)

renderer_p3d = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_soft
    ),
    shader=SoftPhongShader(cameras=cameras,),
)


# Render the 3D model with the texture atlas applied
images = renderer_p3d( mesh)

# normalize the tensor values between 0 and 1
tensor_normalized = (images - images.min()) / (images.max() - images.min())


# convert the tensor to a PIL image
image = transforms.ToPILImage()(tensor_normalized.squeeze()[..., :3].permute(2, 0, 1))

# save the image
image.save(root+'image.png')
# print(f_uv)