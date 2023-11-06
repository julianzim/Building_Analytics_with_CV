import sys
import os
import json
import requests
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def resizeImage(image, new_size):
    image = Image.fromarray(image)
    image = image.resize(new_size)
    return image


def prepareInput(image):
    image = np.array(image)
    image = image.astype(np.float32) / 255
    image = image[None, ...]
    print('Input preparation completed')
    return image


def saveOutput(orig, out, out_fpath=None):
    out_fname = 'output.png'
    plt.imshow(orig, interpolation='none')
    plt.imshow(out, 'jet', interpolation='none', alpha=0.7)
    plt.axis('off')
    if not out_fpath:
        out_fpath = out_fname
    plt.savefig(os.path.join(out_fpath, out_fname), bbox_inches='tight', pad_inches=0)
    print(f'Output saved to {os.path.join("D:/deploy", out_fpath)}')


IMG_FPATH = sys.argv[1]
OUT_FPATH = None
if sys.argv[2]:
    OUT_FPATH = sys.argv[2]

img = np.array(Image.open(IMG_FPATH))

orig = resizeImage(img, (256, 256))
inp = prepareInput(orig)

requests_data = json.dumps({
    'signature_name': 'serving_default',
    'instances': inp.tolist()
})

headers = {'content_type': 'application/json'}

json_responce = requests.post(
    'http://localhost:8501/v1/models/model/versions/1:predict',
    data=requests_data,
    headers=headers
)

prediction = json.loads(json_responce.text)['predictions']
prediction = np.array(prediction)
seg_map = (prediction[0] > 0.5).astype(np.float32)

saveOutput(orig=orig, out=seg_map, out_fpath=OUT_FPATH)

print('Done!')