import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from datetime import datetime
import threading  
import time       
import math
import argparse

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from tqdm import tqdm

from dataset import FontObjDataset
from models import AdvancedUNetGenerator, Discriminator, compute_wgan_gp_loss

class PunctuationIdentityDataset(Dataset):
    def __init__(self, source_font="source.ttf", img_size=256):
        self.samples = []
        punctuations = ['，', '。', '！', '？', '、', '；', '：', ',', '.', '!', '?', ';', ':']
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        for char in punctuations:
            img = Image.new('L', (img_size, img_size), color=255)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(source_font, int(img_size * 0.8))
                bbox = draw.textbbox((0, 0), char, font=font)
                if bbox is not None:
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text(((img_size-w)/2, (img_size-h)/2 - bbox[1]), char, font=font, fill=0)
            except Exception:
                continue
            
            tensor_img = transform(img)
            for label_idx in range(5):
                # 修复：把普通的 int 包装成了 PyTorch 规定的 Tensor 格式
                self.samples.append((tensor_img, tensor_img, torch.tensor(label_idx, dtype=torch.long)))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def gpu_monitor_background(interval_seconds=20,
                           log_filename="gpu_log.csv",
                           device_info="Unknown"):

    header = (
        "timestamp,GPU_ID,name,"
        "utilization.gpu [%],"
        "utilization.memory [%],"
        "memory.used [MiB],"
        "memory.free [MiB],"
        "temperature.gpu [°C]\n"
    )

    if not os.path.exists(log_filename):
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(header)

    while True:

        cmd = (
            "nvidia-smi "
            "--query-gpu="
            "timestamp,"
            "index,"
            "name,"
            "utilization.gpu,"
            "utilization.memory,"
            "memory.used,"
            "memory.free,"
            "temperature.gpu "
            "--format=csv,noheader,nounits"
        )

        try:

            result = os.popen(cmd).read()

            with open(log_filename, "a", encoding="utf-8") as f:

                for line in result.strip().split("\n"):

                    if not line.strip():
                        continue

                    fields = [x.strip() for x in line.split(",")]

                    if len(fields) == 8:

                        formatted = (
                            f"{fields[0]}, "
                            f"{fields[1]}, "
                            f"{fields[2]}, "
                            f"{fields[3]} %, "
                            f"{fields[4]} %, "
                            f"{fields[5]} MiB, "
                            f"{fields[6]} MiB, "
                            f"{fields[7]} °C\n"
                        )

                        f.write(formatted)

        except Exception as e:

            with open(log_filename, "a") as f:
                f.write(f"ERROR,{str(e)}\n")

        time.sleep(interval_seconds)
        
def log_verbose_gpu_snapshot(log_filename):
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"\n=== GPU Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        try:
            detailed_info = os.popen("nvidia-smi").read()
            f.write(detailed_info + "\n")
        except: 
            f.write("Error getting nvidia-smi status\n")

def run_verbose_gpu_monitor(log_filename, interval_seconds=30):
    while True:
        log_verbose_gpu_snapshot(log_filename)
        time.sleep(interval_seconds)

def run_specific_subtask(task_name, use_skip, use_gan, use_vae, loss_type, epochs=100, device='cpu', student_id="2575736"):
    # 使用 tqdm.write 替换 print，防止破坏进度条的固定排版
    tqdm.write(f"\nStarting academic task: {task_name}")
    tqdm.write(f"Config: Skip={use_skip} | GAN={use_gan} | VAE={use_vae} | Loss={loss_type}")

    batch_size = 16
    obj_file = f"{student_id}-FontData.obj"
    
    if not os.path.exists(obj_file):
        tqdm.write(f"Error: Dataset package {obj_file} not found. Please run preprocess.py first.")
        return

    dataset = FontObjDataset(obj_file)
    punc_dataset = PunctuationIdentityDataset()
    combined_dataset = ConcatDataset([dataset, punc_dataset])
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    generator = AdvancedUNetGenerator(in_channels=1, out_channels=1, num_fonts=5, use_skip=use_skip, use_vae=use_vae).to(device)
    opt_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    if use_gan:
        discriminator = Discriminator(in_channels=2).to(device)
        opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

    criterion_recon = nn.L1Loss() if loss_type == 'L1' else nn.MSELoss()

    weight_dir = f"weights_{task_name}"
    os.makedirs(weight_dir, exist_ok=True)
    log_csv = f"train_log_{task_name}.csv"
    
    run_dir = f"runs/{task_name}_{datetime.now().strftime('%m%d_%H%M')}"
    
    os.makedirs(run_dir, exist_ok=True)
    
    threading.Thread(
        target=run_verbose_gpu_monitor, 
        args=(os.path.join(run_dir, "gpu_verbose.txt"), 20), 
        daemon=True
    ).start()

    writer = SummaryWriter(log_dir=run_dir)

    headers = ['Epoch', f'{loss_type}_Recon_Loss']
    if use_gan: headers.extend(['D_Loss', 'G_Adv_Loss'])
    if use_vae: headers.append('KL_Loss') 
        
    with open(log_csv, mode='w', newline='') as f:
        csv.writer(f).writerow(headers)

    best_loss = float('inf')

    bar_pos = device.index if (device.type == 'cuda' and device.index is not None) else 0
    pbar = tqdm(range(epochs), desc=task_name, position=bar_pos, leave=True, ncols=120)

    for epoch in pbar:
        recon_loss_total = 0.0
        d_loss_total = 0.0
        g_adv_loss_total = 0.0
        kl_loss_total = 0.0
        num_batches = 0

        for i, (source_imgs, target_imgs, labels) in enumerate(dataloader):
            source_imgs, target_imgs, labels = source_imgs.to(device), target_imgs.to(device), labels.to(device)
            num_batches += 1

            d_loss = torch.tensor(0.0, device=device)
            g_adv_loss = torch.tensor(0.0, device=device)

            if use_gan:
                opt_D.zero_grad()
                fake_imgs, _ = generator(source_imgs, labels)
                gp = compute_wgan_gp_loss(discriminator, target_imgs, fake_imgs.detach(), source_imgs, device)
                real_validity = discriminator(source_imgs, target_imgs)
                fake_validity = discriminator(source_imgs, fake_imgs.detach())
                critic_gap = -torch.mean(real_validity) + torch.mean(fake_validity)
                d_loss = critic_gap + 10 * gp
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                opt_D.step()
            else:
                critic_gap = torch.tensor(0.0, device=device)

            opt_G.zero_grad()
            fake_imgs, kl_loss = generator(source_imgs, labels)
            g_recon_loss = criterion_recon(fake_imgs, target_imgs)
            
            recon_weight = 800.0 if use_gan else 100.0
            g_loss = recon_weight * g_recon_loss 
            
            kl_val = 0.0
            if use_vae and kl_loss is not None:
                beta = 0.1  
                weighted_kl = kl_loss * beta
                g_loss += weighted_kl
                kl_val = kl_loss.item()

            if use_gan:
                fake_validity = discriminator(source_imgs, fake_imgs)
                g_adv_loss = -torch.mean(fake_validity)
                adv_weight = 5.0 if loss_type == 'L2' else 1.0
                g_loss += adv_weight * g_adv_loss 

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            opt_G.step()

            recon_loss_total += g_recon_loss.item()
            if use_gan:
                # Log the critic gap rather than the full WGAN-GP objective.
                # The gradient penalty is an optimizer stabilizer; including it
                # in the plotted D_Loss can create misleading one-epoch spikes.
                d_loss_total += critic_gap.item()
                g_adv_loss_total += g_adv_loss.item()
            if use_vae:
                kl_loss_total += kl_val

        avg_recon_loss = recon_loss_total / max(1, num_batches)
        avg_d_loss = d_loss_total / max(1, num_batches)
        avg_g_adv_loss = g_adv_loss_total / max(1, num_batches)
        avg_kl_loss = kl_loss_total / max(1, num_batches)

        writer.add_scalar(f'Loss/{loss_type}_Recon', avg_recon_loss, epoch)
        if use_gan:
            writer.add_scalar('Loss/Discriminator', avg_d_loss, epoch)
            writer.add_scalar('Loss/Generator_Adv', avg_g_adv_loss, epoch)
        if use_vae:
            writer.add_scalar('Loss/KL_Divergence', avg_kl_loss, epoch)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                n_vis = min(4, source_imgs.size(0)) 
                src_vis = (source_imgs[:n_vis] * 0.5 + 0.5).cpu()
                tgt_vis = (target_imgs[:n_vis] * 0.5 + 0.5).cpu()
                fak_vis = (fake_imgs[:n_vis].detach() * 0.5 + 0.5).cpu()
                img_grid = vutils.make_grid(torch.cat([src_vis, tgt_vis, fak_vis], dim=0), nrow=n_vis, padding=2)
                writer.add_image('Visual_Progress/Source_vs_Target_vs_Fake', img_grid, epoch)

        # ==========================================
        # 原位更新进度条，不往下刷屏
        # ==========================================
        row_data = [epoch + 1, avg_recon_loss]
        postfix_dict = {f'{loss_type}': f"{avg_recon_loss:.4f}"}
        
        if use_gan:
            row_data.extend([avg_d_loss, avg_g_adv_loss])
            postfix_dict['D'] = f"{avg_d_loss:.4f}"
            postfix_dict['G'] = f"{avg_g_adv_loss:.4f}"
        if use_vae:
            row_data.append(avg_kl_loss)
            postfix_dict['KL'] = f"{avg_kl_loss:.4f}"
            
        pbar.set_postfix(postfix_dict)
        
        with open(log_csv, mode='a', newline='') as f:
            csv.writer(f).writerow(row_data)

        if (epoch + 1) in [int(epochs * 0.3), int(epochs * 0.6), epochs]:
            torch.save(generator.state_dict(), f"{weight_dir}/generator_epoch_{epoch+1}.pth")

        current_loss = avg_recon_loss
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(generator.state_dict(), f"{weight_dir}/generator_best.pth")

    writer.close()

def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def compute_ssim(img1, img2, window_size=11, size_average=True):
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def build_balanced_eval_subset(dataset, samples_per_font=40, num_fonts=5):
    buckets = {label: [] for label in range(num_fonts)}
    for idx in range(len(dataset)):
        _, _, label = dataset[idx]
        label = int(label.item()) if hasattr(label, "item") else int(label)
        if label in buckets and len(buckets[label]) < samples_per_font:
            buckets[label].append(idx)
        if all(len(items) >= samples_per_font for items in buckets.values()):
            break
    indices = []
    for label in range(num_fonts):
        indices.extend(buckets[label])
    if not indices:
        raise RuntimeError("Evaluation subset is empty. Check the .obj dataset labels.")
    return Subset(dataset, indices), {label: len(items) for label, items in buckets.items()}

def sobel_edges(imgs):
    kernel_x = torch.tensor(
        [[[[-1., 0., 1.],
           [-2., 0., 2.],
           [-1., 0., 1.]]]],
        device=imgs.device,
        dtype=imgs.dtype
    ) / 4.0
    kernel_y = torch.tensor(
        [[[[-1., -2., -1.],
           [0., 0., 0.],
           [1., 2., 1.]]]],
        device=imgs.device,
        dtype=imgs.dtype
    ) / 4.0
    grad_x = F.conv2d(imgs, kernel_x, padding=1)
    grad_y = F.conv2d(imgs, kernel_y, padding=1)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)

def laplacian_response(imgs):
    kernel = torch.tensor(
        [[[[0., 1., 0.],
           [1., -4., 1.],
           [0., 1., 0.]]]],
        device=imgs.device,
        dtype=imgs.dtype
    )
    return F.conv2d(imgs, kernel, padding=1)

def compute_batch_calligraphy_metrics(fake_imgs, target_imgs):
    batch_size = fake_imgs.size(0)

    l1_error = F.l1_loss(fake_imgs, target_imgs, reduction="mean").item()
    mse_error = F.mse_loss(fake_imgs, target_imgs, reduction="mean").item()
    ssim_score = compute_ssim(fake_imgs, target_imgs).item()

    fake_ink = (fake_imgs < -0.5).float()
    target_ink = (target_imgs < -0.5).float()
    intersection = (fake_ink * target_ink).flatten(1).sum(dim=1)
    union = fake_ink.flatten(1).sum(dim=1) + target_ink.flatten(1).sum(dim=1) - intersection
    ink_iou = (intersection / (union + 1e-8)).mean().item()
    ink_density_gap = (fake_ink.flatten(1).mean(dim=1) - target_ink.flatten(1).mean(dim=1)).abs().mean().item()

    edge_l1 = F.l1_loss(sobel_edges(fake_imgs), sobel_edges(target_imgs), reduction="mean").item()

    fake_lap = laplacian_response(fake_imgs)
    target_lap = laplacian_response(target_imgs)
    fake_detail = fake_lap.flatten(1).var(dim=1, unbiased=False)
    target_detail = target_lap.flatten(1).var(dim=1, unbiased=False)
    stroke_detail_ratio = (fake_detail / (target_detail + 1e-8)).clamp(max=3.0).mean().item()

    return {
        "samples": batch_size,
        "l1_error": l1_error,
        "mse_error": mse_error,
        "ssim": ssim_score,
        "ink_iou": ink_iou,
        "ink_density_gap": ink_density_gap,
        "edge_l1": edge_l1,
        "stroke_detail_ratio": stroke_detail_ratio,
    }

def finalize_calligraphy_score(metrics):

    final_score = (
        0.30 * metrics["mse_error"]
        + 0.45 * (1.0 - metrics["ink_iou"])
        + 0.20 * (1.0 - metrics["ssim"])
        + 0.05 * metrics["edge_l1"]
    )

    return final_score

def evaluate_and_export_best(device, student_id="2575736"):
    print(f"\nStarting automated unified evaluation and report generation...")
    
    obj_file = f"{student_id}-FontData.obj"
    if not os.path.exists(obj_file):
        print(f"Error: Cannot find {obj_file}, evaluation aborted.")
        return

    dataset = FontObjDataset(obj_file)
    eval_dataset, label_counts = build_balanced_eval_subset(dataset, samples_per_font=40, num_fonts=5)
    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    task_configs = {
        "Task1_Baseline": {"skip": True,  "vae": False, "desc": "U-Net + Skip + L1"},
        "Task2_Ablation": {"skip": False, "vae": False, "desc": "Bottleneck (No Skip) + L1"},
        "Task3_VAE":      {"skip": True,  "vae": True,  "desc": "U-Net + Skip + VAE + L1"},
        "Task4_GAN_L1":   {"skip": True,  "vae": False, "desc": "U-Net + Skip + GAN + L1"},
        "Task5_GAN_L2":   {"skip": True,  "vae": False, "desc": "U-Net + Skip + GAN + L2"}
    }
    
    results = []
    best_task = None
    best_loss = float('inf')
    loaded_models = 0
    
    print(f"Balanced evaluation samples per font: {label_counts}")
    print("Performing calligraphy-oriented evaluation (MSE + Ink IoU + SSIM + Edge + Stroke Detail).")
    
    with torch.no_grad():
        for task_name, cfg in task_configs.items():
            weight_path = f"weights_{task_name}/generator_epoch_100.pth"
            if not os.path.exists(weight_path):
                print(f"Missing checkpoint: {weight_path}, skipping task.")
                continue

            loaded_models += 1
                
            generator = AdvancedUNetGenerator(in_channels=1, out_channels=1, num_fonts=5, 
                                              use_skip=cfg['skip'], use_vae=cfg['vae']).to(device)
            generator.load_state_dict(torch.load(weight_path, map_location=device))
            generator.eval()

            metric_sums = {
                "l1_error": 0.0,
                "mse_error": 0.0,
                "ssim": 0.0,
                "ink_iou": 0.0,
                "ink_density_gap": 0.0,
                "edge_l1": 0.0,
                "stroke_detail_ratio": 0.0,
            }
            total_samples = 0

            for source_imgs, target_imgs, labels in dataloader:
                source_imgs = source_imgs.to(device)
                target_imgs = target_imgs.to(device)
                labels = labels.to(device)

                fake_imgs, _ = generator(source_imgs, labels)
                batch_metrics = compute_batch_calligraphy_metrics(fake_imgs, target_imgs)
                batch_size = batch_metrics["samples"]
                total_samples += batch_size

                for key in metric_sums:
                    metric_sums[key] += batch_metrics[key] * batch_size

            metrics = {key: value / total_samples for key, value in metric_sums.items()}
            eval_loss = finalize_calligraphy_score(metrics)
            results.append({
                "name": task_name,
                "config": cfg['desc'],
                "score": eval_loss,
                **metrics
            })
            print(
                f"  -> {task_name} | Final: {eval_loss:.5f} | MSE: {metrics['mse_error']:.5f} | "
                f"InkIoU: {metrics['ink_iou']:.5f} | SSIM: {metrics['ssim']:.5f} | "
                f"EdgeL1: {metrics['edge_l1']:.5f} | StrokeDetail: {metrics['stroke_detail_ratio']:.5f}"
            )
            
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_task = task_name
                
    if loaded_models == 0:
        print("\nError: No trained model checkpoints found. Please train first.")
        return
        
    if best_task:
        results.sort(key=lambda item: item["score"])

        with open("best_model_config.txt", "w") as f:
            f.write(best_task)

        with open("model_evaluation_metrics.csv", "w", newline="") as f:
            fieldnames = [
                "Task", "Config", "Final_Score",
                "L1_Error", "MSE_Error", "SSIM", "Ink_IoU", "Ink_Density_Gap",
                "Edge_L1", "Stroke_Detail_Ratio"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                writer.writerow({
                    "Task": res["name"],
                    "Config": res["config"],
                    "Final_Score": res["score"],
                    "L1_Error": res["l1_error"],
                    "MSE_Error": res["mse_error"],
                    "SSIM": res["ssim"],
                    "Ink_IoU": res["ink_iou"],
                    "Ink_Density_Gap": res["ink_density_gap"],
                    "Edge_L1": res["edge_l1"],
                    "Stroke_Detail_Ratio": res["stroke_detail_ratio"],
                })
            
        with open("model_leaderboard.txt", "w") as f:
            f.write("Experiment Leaderboard\n")
            f.write(f"Student ID: {student_id}\n\n")
            f.write(f"{'Task Name':<20} | {'Architecture & Loss Setup':<30} | {'Score'}\n")
            f.write("-" * 65 + "\n")
            for res in results:
                winner_tag = " (Optimal Selection)" if res['name'] == best_task else ""
                f.write(f"{res['name']:<20} | {res['config']:<30} | {res['score']:.6f}{winner_tag}\n")
            f.write("-" * 65 + "\n")
            f.write(f"Final Selection: {best_task}\n")
            f.write("Selection Logic: Balanced calligraphy score = 0.30*MSE + 0.45*(1-InkIoU) + 0.20*(1-SSIM) + 0.05*EdgeL1.\n")
            f.write("Detailed Metrics: model_evaluation_metrics.csv\n")

        print(f"\nSelection complete. Optimal configuration: {best_task}")

def run_training_pipeline(args, device, student_id):

    print("\nTraining Phase Started")

    task_list = [
        ("Task1_Baseline", True, False, False, "L1"),
        ("Task2_Ablation", False, False, False, "L1"),
        ("Task3_VAE", True, False, True, "L1"),
        ("Task4_GAN_L1", True, True, False, "L1"),
        ("Task5_GAN_L2", True, True, False, "L2"),
    ]

    if len(args.tasks) > 0:
        task_list = [t for t in task_list if t[0] in args.tasks]

    for task_name, use_skip, use_gan, use_vae, loss_type in task_list:

        final_weight = f"weights_{task_name}/generator_epoch_100.pth"

        if os.path.exists(final_weight):
            print(f"\nSkipping {task_name}: already trained")
            continue

        run_specific_subtask(
            task_name=task_name,
            use_skip=use_skip,
            use_gan=use_gan,
            use_vae=use_vae,
            loss_type=loss_type,
            epochs=100,
            device=device,
            student_id=student_id
        )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help="List of tasks to run"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval"
    )

    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    print("\nStarting Academic Training Pipeline")
    print(f"Device: {device}")

    device_info = str(device).upper()

    if device.type == "cuda":

        gpu_name = torch.cuda.get_device_name(args.gpu)

        device_info = f"GPU {args.gpu} ({gpu_name})"

        monitor = threading.Thread(
            target=gpu_monitor_background,
            args=(20, "gpu_log.csv", device_info)
        )

        monitor.daemon = True
        monitor.start()

        print(f"GPU Monitoring Enabled: {device_info}")

    elif device.type == "mps":

        print("Apple MPS acceleration enabled")

    else:

        print("Warning: Running on CPU")

    student_id = "2575736"

    obj_file = f"{student_id}-FontData.obj"

    if not os.path.exists(obj_file):

        print(f"\nError: Dataset not found: {obj_file}")

        return

    print(f"Dataset Loaded: {obj_file}")

    if args.mode == "train":

        print(f"\n[*] Training Mode | GPU {args.gpu}")

        run_training_pipeline(
            args,
            device,
            student_id
        )

    elif args.mode == "eval":

        print("\n[*] Evaluation Mode")

        evaluate_and_export_best(
            device=device,
            student_id=student_id
        )

    print("\nAll academic tasks successfully completed.")


if __name__ == "__main__":
    main()
