import sanghyunjo as shjo

from PIL import Image
from torchvision import transforms

from networks import StableDiffusionTurbo

model = StableDiffusionTurbo()

model.load('./last_100000.pt')
model.eval()

input_dir = './data/rgb_results/'
output_dir = shjo.makedir('./results_test/')

for image_name in shjo.listdir(input_dir):
    if not '.jpg' in image_name: #.JPG
        continue

    image = Image.open(input_dir + image_name)
    image = image.resize((512,512))

    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)

    image_rec = model.forward(image[None].cuda(), "")[0] 

    import numpy as np
    image_rec = image_rec * 0.5 + 0.5
    image_rec = (image_rec.cpu().detach().numpy() * 255).astype(np.uint8)
    image_rec = image_rec.transpose((1, 2, 0))

    Image.fromarray(image_rec).save(output_dir + image_name)

