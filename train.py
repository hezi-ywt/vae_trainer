import os
import logging
import time

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
logger = logging.getLogger("hz_sdxl_vae_trainer")

accelerator = Accelerator(mixed_precision='fp16')

# 指定VAE的路径和参数类型
vae_id = "/home1/qbs/HuggingFace/sdxl-vae-fp16-fix/sdxl_vae.safetensors"
dtype = "float16"
batch_size = 1
num_epochs = 10
lr = 4e-5
accumulation_steps = 16 # 根据你的显存情况可以调整
args = None
resolution = 1024
random_flip = True
index_file = "dataset/porcelain/jsons/porcelain_mt.json"
multireso = True
global_seed = 114514
num_workers = 4
save_every_step = 100

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
    transforms.ToTensor(),
])

dataset = TextImageArrowStream(args=args,
                               resolution=resolution,
                               random_flip=random_flip,
                               log_fn=logger.info,
                               index_file=index_file,
                               multireso=multireso,
                               batch_size=batch_size,
                               world_size=accelerator.num_processes
                               )

if multireso:
    sampler = BlockDistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, seed=global_seed,
                                      shuffle=False, drop_last=True, batch_size=batch_size)
else:
    sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, seed=global_seed,
                                               shuffle=False, drop_last=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
logger.info(f"Dataset contains {len(dataset):,} images.")
logger.info(f"Index file: {index_file}.")
if multireso:
    logger.info(f'Using MultiResolutionBucketIndexV2 with step {dataset.index_manager.step} '
                f'and base size {dataset.index_manager.base_size}')
    logger.info(f'\n{dataset.index_manager.resolutions}')

# 设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.decoder.parameters(), lr=lr)

vae, optimizer, train_loader = accelerator.prepare(vae, optimizer, train_loader)

logger.info("Training started.")

global_step = 0

def format_time(seconds):
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # 记录 epoch 开始时间
    batch_start_time = epoch_start_time  # 记录第一个 batch 开始时间

    vae.train()
    running_loss = 0.0
    for batch_idx, images in enumerate(train_loader):
        global_step += 1
        images = images[0]

        images = images.to(accelerator.device, dtype=torch.float16 if dtype == "float16" else torch.float32)

        with accelerator.autocast():
            # 通过unwrap_model来调用原始模型的自定义方法
            original_vae = accelerator.unwrap_model(vae)
            latents = original_vae.encode(images)["latent_dist"].sample()

            # 解码
            reconstructed_images = original_vae.decode(latents).sample

            # 计算损失
            loss = criterion(reconstructed_images, images)  # 使用自定义损失函数

        # 反向传播累加梯度
        loss = loss / accumulation_steps
        accelerator.backward(loss)

        # 每 accumulation_steps 步进行一次优化更新和梯度清零
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            accelerator.print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Step [{global_step}], "
            f"Batch [{batch_idx + 1}/{len(train_loader)}], "
            f"Loss: {loss.item() * accumulation_steps:.4f}, "
            f"Batch Time: {batch_time:.2f}s, "
            f"Elapsed: {formatted_elapsed_time}, "
            f"ETA: {formatted_remaining_time}")

        running_loss += loss.item() * accumulation_steps

        # 每步记录和更新时间
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        elapsed_time = batch_end_time - epoch_start_time
        remaining_time = (len(train_loader) - batch_idx - 1) * batch_time
        batch_start_time = batch_end_time

        # 格式化时间
        formatted_elapsed_time = format_time(elapsed_time)
        formatted_remaining_time = format_time(remaining_time)



        # 完成每1000步后保存模型状态
        if global_step % save_every_step == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                state_dict = accelerator.unwrap_model(vae).state_dict()
                os.makedirs(f"output_{batch_size * accumulation_steps}_{lr}", exist_ok=True)
                save_file(state_dict, f"output_{batch_size * accumulation_steps}_{lr}/vae_trainer_step_{global_step}.safetensors")
                accelerator.print(f"Saved model at global step {global_step}")


    epoch_time = time.time() - epoch_start_time  # 计算 epoch 花费时间
    logger.info(f"Epoch [{epoch + 1}/{num_epochs}] completed with Loss: {running_loss / len(train_loader):.4f}, Total Time: {format_time(epoch_time)}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 保存每个epoch结束时模型状态
        state_dict = accelerator.unwrap_model(vae).state_dict()
        os.makedirs(f"output_{batch_size * accumulation_steps}_{lr}", exist_ok=True)
       

logger.info("Training Complete.")
