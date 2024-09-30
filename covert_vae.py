import os
import logging
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator

from model import load_vae
from safetensors.torch import load_file

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

accelerator = Accelerator(mixed_precision='fp16')

def test_and_generate_images(vae, transform, test_data_path, results_path, model_name, dtype="float16"):
    test_dataset = datasets.ImageFolder(test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    vae, test_loader = accelerator.prepare(vae, test_loader)
    vae.eval()

    with accelerator.autocast():
        for batch_idx, (images, _) in enumerate(test_loader):
            
            images = images.to(accelerator.device, dtype=torch.float16 if dtype == "float16" else torch.float32)
            
            latents = vae.encode(images)["latent_dist"].sample()
            reconstructed_images = vae.decode(latents).sample

            # 将图像从GPU移回CPU
            images = images.cpu()
            reconstructed_images = reconstructed_images.cpu()

            # 绘制并保存对比图
            for i in range(images.size(0)):
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(images[i].permute(1, 2, 0))
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(reconstructed_images[i].permute(1, 2, 0))
                axes[1].set_title(f'{model_name} Reconstructed')
                axes[1].axis('off')

                plt.savefig(os.path.join(results_path, f'{model_name}_compare_batch_{batch_idx+1}_image_{i+1}.png'))
                plt.close(fig)

    logger.info(f"Test and figure generation for {model_name} complete.") 

def main():
    vae_path = "/home/ywt/lab/stable-diffusion-webui/models/VAE/sdxl.vae.safetensors"  # 修改为实际的模型路径
    checkpoint_paths = [
        "/home/ywt/lab/stable-diffusion-webui/models/VAE/sdxl.vae.safetensors",  # 原始模型检查点路径
        "/home/ywt/lab/sd-scripts/library/vae_trainer/output/vae_trainer_epoch_1.safetensors",  # 修改为不同检查点路径
        "/home/ywt/lab/sd-scripts/library/vae_trainer/output/vae_trainer_epoch_2.safetensors",
        "/home/ywt/lab/sd-scripts/library/vae_trainer/output/vae_trainer_epoch_3.safetensors"
    ]
    model_names = [
        "Original",
        "Trained_epoch_1",
        "Trained_epoch_2",
        "Trained_epoch_3"
    ]
    test_data_path = "/mnt/e/HYV2/test_images"
    results_path = "/home/ywt/lab/sd-scripts/library/vae_trainer/output_image"  # 修改为实际的结果保存路径

    os.makedirs(results_path, exist_ok=True)

    dtype = "float16"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for checkpoint_path, model_name in zip(checkpoint_paths, model_names):
        # 加载VAE模型
        vae = load_vae(checkpoint_path, dtype)
        
        # # 加载检查点文件
        # state_dict = load_file(checkpoint_path)
        # vae.load_state_dict(state_dict)
        vae.to(accelerator.device)

        # 测试并生成对比图
        test_and_generate_images(vae, transform, test_data_path, results_path, model_name, dtype)

if __name__ == "__main__":
    vae1 = load_file("/home/ywt/lab/stable-diffusion-webui/models/VAE/vae_trainer_epoch_1.safetensors")
    from model import convert_vae_state_dict
    vae1 = convert_vae_state_dict(vae1)
    from safetensors.torch import save_file
    save_file(vae1, "/home/ywt/lab/stable-diffusion-webui/models/VAE/vae_trainer_epoch_1.safetensors")
    