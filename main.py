from Generator import Generator
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
nz = 100

generator = Generator(ngpu = 0)
generator.load_state_dict(torch.load('generator.pth'))
generator = generator.to(device)
generator.eval()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
with torch.no_grad():
    fake = generator(fixed_noise).detach().cpu()
    fake = fake[0]

save_image(fake, 'fake.png')
img = Image.open('fake.png')
print(img.size)
p = transforms.Compose([transforms.Scale((300,300))])
img = p(img)
print(img)
img.save('fake.png')