import open3d as o3d
from PIL import Image
from skimage.io import imread, imsave
import numpy as np
from ldm.util import add_margin

### Mesh Visualization ###
def mesh_visualization(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


### Image Split ###
def split_image(input_image, output_path):
    img_list = imread(input_image)
    multiview_list = np.array_split(img_list, 4, 0)
    for i, tmp_mv_list in enumerate(multiview_list):
        img_list = np.array_split(tmp_mv_list, 16, 1)
        for j, img in enumerate(img_list):
            imsave(f'{output_path}/{i}_{j}.png', img)

### Resize Image ###
def make_squared_img(input_image):
    crop_size=240
    image_size=256

    img = Image.open(input_image)
    alpha_np = np.asarray(img)[:, :, 3]
    coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
    min_x, min_y = np.min(coords, 0)
    max_x, max_y = np.max(coords, 0)
    ref_img_ = img.crop((min_x, min_y, max_x, max_y))
    h, w = ref_img_.height, ref_img_.width
    scale = crop_size / max(h, w)
    h_, w_ = int(scale * h), int(scale * w)
    ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
    rename_input = add_margin(ref_img_, size=image_size)
    resize_img = input_image.replace('.png', '_resize.png')
    imsave(resize_img, np.asarray(rename_input))

