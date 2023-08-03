import os
import io
import warnings

from PIL import Image
import PIL.ImageOps    
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from torchvision.transforms import GaussianBlur

from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime


# Our host url should not be prepended with "https" nor should it have a trailing slash.
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

# Sign up for an account at the following link to get an API Key.
# https://platform.stability.ai/

# Click on the following link once you have created an account to be taken to your API Key.
# https://platform.stability.ai/account/keys

# Paste your API Key below.

os.environ['STABILITY_KEY'] = 'sk-5zhbtQ9HfrQLC2jCybQLdnLHPkdfqNEjUtt8maHtIY9SkyeL'


# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-xl-1024-v1-0", # Set the engine to use for generation.
    # Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
)

root_path = '/Users/bytedance/projects/stabilityai-inpainting/'
image_path = root_path+'image.png'
mask_path = root_path+'mask.png'
    
def run_logic(img, mask_i, prompt):
    max_size = 1024
    # resize and prepare
    aspect_ratio = float(img.size[0])/img.size[1]
    img = img.resize((max_size, max_size))
    img.save(root_path+'image_resized.png')
    mask_i = PIL.ImageOps.invert(mask_i)
    mask_i = mask_i.resize(img.size)
    mask_i.save(root_path+'mask_resized.png')


    # Feathering the edges of our mask generally helps provide a better result. Alternately, you can feather the mask in a suite like Photoshop or GIMP.
    blur = GaussianBlur(11,20)
    mask = blur(mask_i)

    answers = stability_api.generate(
        # prompt="crayon drawing of rocket ship launching from forest",
        prompt=prompt,
        init_image=img,
        mask_image=mask,
        start_schedule=1,
        seed=44332211, # If attempting to transform an image that was previously generated with our API,
                    # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
        steps=50, # Amount of inference steps performed on image generation. Defaults to 30.
        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=max_size, # Generation width, if not included defaults to 512 or 1024 depending on the engine.
        height=max_size, # Generation height, if not included defaults to 512 or 1024 depending on the engine.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                    # Defaults to k_lms if not specified. Clip Guidance only supports ancestral samplers.
                                                    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated image.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img2
                img2 = Image.open(io.BytesIO(artifact.binary))
                img2 = img2.resize((int(max_size*aspect_ratio), max_size))
                img2.save(root_path+str(artifact.seed)+ ".png") # Save our completed image with its seed number as the filename.


# server API
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = root_path + "uploads/"

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file(key):
    # check if the post request has the file part
    if key not in request.files:
        print('No file part')
        return None
    file = request.files[key]
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        print('No selected file')
        return None
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        return path
    return None

@app.route("/", methods=['POST'])
def main_route():
    print("hit main route")

    # check if the post request has the file part
    image = get_file('image')
    mask = get_file('mask')
    prompt = request.form.get('prompt')

    if image and mask and prompt:
        img = Image.open(image)
        mask_i = Image.open(mask)
        start = datetime.now()
        run_logic(img, mask_i, prompt)
        print("generation success, time taken = {}".format((datetime.now() - start).total_seconds()))

    return "success"

if __name__ == '__main__':
   app.run(debug=True, port=8080)