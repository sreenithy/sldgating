import sys
import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import cv2
import joblib
import tqdm

# num samples per pixel
pixel_sample = 16

# render image size.
render_width = 256
render_height = 256

# filename to be rendered
basefilename = "scene/template.pbrt"

# projection image resolution
projection_height = 8
projection_width = 8


def parser():
    usage = "Usage: python {} FILE [-p/--parallel]".format(__file__)
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument(
        "--pbrt_path",
        type=str,
        dest="pbrt_path",
        default="/opt/pbrt-v3/build/pbrt",
        help="path to PBRT v3 executable",
    )
    argparser.add_argument(
        "-p",
        "--parallel",
        dest="parallel",
        action="store_true",
        help="enables parallel rendering",
    )
    argparser.add_argument(
        "--tonemap",
        type=str,
        choices=[None, "gamma", "durand"],
        help="algorithm for HDR image tone mapping",
    )
    argparser.add_argument(
        "-o",
        "--outdir",
        type=str,
        dest="outdir",
        default="output",
        help="output directory",
    )

    args = argparser.parse_args()
    return args


def generate_projection_image(index, width, height):
    px = int(index % width)
    py = int((index - px) / width)
    image = np.zeros((projection_height, projection_width), dtype=np.uint8)
    image[py, px] = 255
    return image


def generate_pbrt_scene_file(
    base_scene_file_path,
    output_scene_filename,
    output_image_filename,
    projection_filename,
    pixel_sample,
    render_width,
    render_height,
):
    projection_relative_filename = os.path.relpath(
        projection_filename, os.path.dirname(output_scene_filename)
    )

    with open(base_scene_file_path, "r", encoding="utf-8") as file:
        filedata = file.read()
    filedata = filedata.replace("__PIXEL_SAMPLE__", str(pixel_sample))
    filedata = filedata.replace("__X_RESOLUTION__", str(render_width))
    filedata = filedata.replace("__Y_RESOLUTION__", str(render_height))
    filedata = filedata.replace(
        "__OUTPUT_FILENAME__", output_image_filename.replace("\\", "/")
    )
    filedata = filedata.replace(
        "__PROJECTION_FILENAME__", projection_relative_filename.replace("\\", "/")
    )
    with open(output_scene_filename, "w", encoding="utf-8") as file:
        file.write(filedata)


def convert_exr2png(exr_filename, png_filename, algo=None):
    image_exr = cv2.imread(exr_filename, -1)
    if algo is None:
        # Do nothing
        image_png = image_exr
    elif algo == "gamma":
        # gamma correction
        image_png = np.power(np.clip(image_exr, 0.0, 1.0), 1.0 / 2.2)
    elif algo == "durand":
        tmo = cv2.createTonemap(gamma=2.2)
        image_png = tmo.process(image_exr.copy())
    else:
        raise RuntimeError("Unknown TMO method: %s" % (algo))

    image_png = np.clip((255 * image_png), 0, 255).astype(np.uint8)
    cv2.imwrite(png_filename, image_png)


def render_image(
    pbrt_path, index, num_thread=1, tonemap=None, outdir="output", quiet=False
):
    
    projection_filename = os.path.join(
        "scene", "projection" + str(index) + ".png"
    )  # projection image filename to be generated.
    output_exr_filename = os.path.join(
        outdir, "out" + str(index).zfill(8) + ".exr"
    )  # output image format: exr (can store float value)
    temp_scenefilename = os.path.join("scene", "_scene" + str(index) + ".pbrt")

    os.makedirs(outdir, exist_ok=True)

    image = generate_projection_image(index, projection_width, projection_height)
    cv2.imwrite(projection_filename, image)

    generate_pbrt_scene_file(
        basefilename,
        temp_scenefilename,
        output_exr_filename,
        projection_filename,
        pixel_sample,
        render_width,
        render_height,
    )

    # rendering
    subprocess.call(
        [
            pbrt_path,
            temp_scenefilename,
            "--nthreads=" + str(num_thread),
            ("--quiet" if quiet else ""),
        ]
    )

    # convert rendered image from openexr (float) to png (uint8) for preview
    png_filename = os.path.splitext(output_exr_filename)[0] + ".png"
    convert_exr2png(output_exr_filename, png_filename, algo=tonemap)

    # remove temporary scene file and projection image file
    os.remove(temp_scenefilename)
    os.remove(projection_filename)

    return True


def main(argv):
    print("Hello, World!")
    arg = parser()

    if not arg.parallel:
        # single thread process
        for i in tqdm.tqdm(range(projection_width * projection_height)):
            render_image(
                arg.pbrt_path,
                i,
                num_thread=1,
                tonemap=arg.tonemap,
                outdir=arg.outdir,
                quiet=False,
            )
    else:
        # multi thread process
        result = joblib.Parallel(n_jobs=-1, verbose=10)(
            joblib.delayed(render_image)(
                arg.pbrt_path,
                i,
                num_thread=1,
                tonemap=arg.tonemap,
                outdir=arg.outdir,
                quiet=True,
            )
            for i in range(projection_width * projection_height)
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
