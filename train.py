
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator
from safetensors.torch import save_file 


from IndexKits.index_kits import ResolutionGroup
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler

from model import load_vae
from data_loader.arrow_load_stream import TextImageArrowStream
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


accelerator = Accelerator(mixed_precision='fp16') 

# 指定VAE的路径和参数类型
vae_id = "/home/ywt/lab/stable-diffusion-webui/models/VAE/sdxl.vae.safetensors"
dtype = "float16"
batch_size = 1
args=None
resolution=1024
random_flip=False
index_file="dataset/porcelain/jsons/porcelain_mt.json"
multireso=True
random_shrink_size_cond=False
world_size=1
global_seed=0
rank=0
num_workers=4
# 加载并设置VAE模型
vae = load_vae(vae_id, dtype)
logger.info("VAE Model loaded successfully.")

# 只训练解码器部分
for param in vae.parameters():
    param.requires_grad = False
for param in vae.decoder.parameters():
    param.requires_grad = True

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

# 数据预处理和加载
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
# dataset = datasets.ImageFolder("/mnt/e/HYV2/images", transform=transform)
dataset = TextImageArrowStream(args=args,
                                   resolution=resolution,
                                   random_flip=random_flip,
                                   log_fn=logger.info,
                                   index_file=index_file,
                                   multireso=multireso,
                                   batch_size=batch_size,
                                   world_size=world_size,
                                   random_shrink_size_cond=random_shrink_size_cond,
                                   )


if multireso:
    sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=global_seed,
                                        shuffle=False, drop_last=True, batch_size=batch_size)
else:
    sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=rank, seed=global_seed,
                                                shuffle=False, drop_last=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                    num_workers=num_workers, pin_memory=True, drop_last=True)
logger.info(f"    Dataset contains {len(dataset):,} images.")
logger.info(f"    Index file: {index_file}.")
if multireso:
    logger.info(f'    Using MultiResolutionBucketIndexV2 with step {dataset.index_manager.step} '
                f'and base size {dataset.index_manager.base_size}')
    logger.info(f'\n  {dataset.index_manager.resolutions}')
    
        


# 设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.decoder.parameters(), lr=4.5e-5)


vae, optimizer, train_loader = accelerator.prepare(vae, optimizer, train_loader)


logger.info("Training started.")
# 训练过程
num_epochs = 10

# 梯度累加配置
accumulation_steps = 16  # 根据你的显存情况可以调整
global_step = 0
for epoch in range(num_epochs):
    vae.train()
    running_loss = 0.0
    for batch_idx, images in enumerate(train_loader):
        global_step += 1
        images = images[0]
        
        images = images.to(accelerator.device, dtype=torch.float16 if dtype == "float16" else torch.float32)
        
        with accelerator.autocast():
            # 编码器输出假设是一个分布
            latents = vae.encode(images)["latent_dist"].sample() 
            
            # 解码
            reconstructed_images = vae.decode(latents).sample

            # 计算损失
            loss = criterion(reconstructed_images, images)  # 使用自定义损失函数

        # 反向传播累加梯度
        loss = loss / accumulation_steps
        accelerator.backward(loss)
        
        # 每 accumulation_steps 步进行一次优化更新和梯度清零
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            logger.info(f"Step [{global_step}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item() * accumulation_steps}")
       
        running_loss += loss.item() * accumulation_steps
    
    logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    # 保存模型状态为safetensors文件
    state_dict = accelerator.unwrap_model(vae).state_dict()
    save_file(state_dict, f"/home/ywt/lab/sd-scripts/library/vae_trainer_epoch_{epoch+1}.safetensors")


logger.info("Training Complete.")
