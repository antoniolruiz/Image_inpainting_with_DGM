from google.cloud import storage
import io
from PIL import Image

client = storage.Client()
# https://console.cloud.google.com/storage/browser/[bucket-id]/

bucket = client.get_bucket('inpainting-final-project')
blob = bucket.get_blob('images/Cars/cars_train/00001.jpg')
s = blob.download_as_string()

img = Image.open(io.BytesIO(s))
img