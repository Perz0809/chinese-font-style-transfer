import os
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont

from models import AdvancedUNetGenerator

TASK_CONFIGS = {
    "Task1_Baseline": {"skip": True,  "vae": False},
    "Task2_Ablation": {"skip": False, "vae": False},
    "Task3_VAE":      {"skip": True,  "vae": True},
    "Task4_GAN_L1":   {"skip": True,  "vae": False},
    "Task5_GAN_L2":   {"skip": True,  "vae": False},
}

def render_char(char, font_path, img_size=256):
    img = Image.new('L', (img_size, img_size), color=255)
    if char.isspace():
        return img  # Return blank image for spaces or newlines
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, int(img_size * 0.8))
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((img_size - w) / 2, (img_size - h) / 2 - bbox[1]), char, font=font, fill=0)
        return img
    except Exception:
        return img

def export_tang_poem_matrix(task_name, cfg, clean_chars, source_font, transform, device):
    weight_path = f"weights_{task_name}/generator_epoch_100.pth"
    if not os.path.exists(weight_path):
        print(f"Skipping {task_name}: missing weights at {weight_path}")
        return None

    generator = AdvancedUNetGenerator(
        in_channels=1,
        out_channels=1,
        num_fonts=5,
        use_skip=cfg["skip"],
        use_vae=cfg["vae"],
        compress_bottleneck=False
    ).to(device)

    generator.load_state_dict(torch.load(weight_path, map_location=device))
    generator.eval()

    all_char_tensors = []
    print(f"Generating Tang poem matrix for {task_name}...")

    with torch.no_grad():
        for char in clean_chars:
            src_pil = render_char(char, source_font)
            src_tensor = transform(src_pil).unsqueeze(0).to(device)

            src_display = (src_tensor.cpu().squeeze(0) + 1) / 2
            char_row = [src_display]

            for style_id in range(5):
                label_tensor = torch.tensor([style_id], dtype=torch.long).to(device)
                fake_tensor, _ = generator(src_tensor, label_tensor)
                gen_display = (fake_tensor.cpu().squeeze(0) + 1) / 2
                char_row.append(gen_display)

            all_char_tensors.extend(char_row)

    padding = 4
    img_size = 256
    grid = vutils.make_grid(all_char_tensors, nrow=6, padding=padding, pad_value=1.0)
    
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    grid_img = Image.fromarray(ndarr)
    
    # Render visual headers for academic presentation
    header_h = 80
    final_w = grid_img.width
    final_h = grid_img.height + header_h
    final_canvas = Image.new('L', (final_w, final_h), color=255)
    final_canvas.paste(grid_img, (0, header_h))  # Offset grid downward to allocate header space
    
    draw = ImageDraw.Draw(final_canvas)
    try:
        header_font = ImageFont.truetype(source_font, 36)
    except Exception:
        header_font = ImageFont.load_default()
        
    headers = ["Source (GT)", "Target 0", "Target 1", "Target 2", "Target 3", "Target 4"]
    col_width = img_size + padding
    
    for c_idx, title in enumerate(headers):
        bbox = draw.textbbox((0, 0), title, font=header_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = padding + c_idx * col_width + (img_size - text_w) / 2  # Calculate exact center for each column
        y = (header_h - text_h) / 2 - bbox[1]
        draw.text((x, y), title, font=header_font, fill=0)
    
    output_filename = f"TangPoem_Matrix_by_{task_name}.png"
    final_canvas.save(output_filename)
    print(f"Matrix exported to: {output_filename}")
    return output_filename

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nStarting Multi-Model Tang Poem Inference Pipeline. Device: {device}")

    poem_path = "ExampleTangPoem.txt"
    source_font = "source.ttf"

    if not os.path.exists(poem_path) or not os.path.exists(source_font):
        print("Error: Missing dataset files (poem text or source font).")
        return

    with open(poem_path, 'r', encoding='utf-8') as f:
        poem_content = f.read()

    clean_chars = [c for c in poem_content if not c.isspace()]
    print(f"Test sequence loaded successfully. Total effective characters: {len(clean_chars)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    exported_files = []
    for task_name, cfg in TASK_CONFIGS.items():
        output_file = export_tang_poem_matrix(task_name, cfg, clean_chars, source_font, transform, device)
        if output_file is not None:
            exported_files.append(output_file)

    if os.path.exists("best_model_config.txt"):
        with open("best_model_config.txt", "r") as f:
            winner_task = f.read().strip()
        selected_image = f"TangPoem_Matrix_by_{winner_task}.png"
        best_image = f"BEST_TangPoem_FinalSelection_{winner_task}.png"

        if os.path.exists(selected_image):
            shutil.copyfile(selected_image, best_image)

        print(f"Final selected model from leaderboard: {winner_task}")
        print(f"Final selected Tang poem image: {best_image}")

    print(f"Generated {len(exported_files)} Tang poem matrix images.")

if __name__ == "__main__":
    main()
