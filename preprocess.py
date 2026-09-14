import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Extract base directory as a global constant to resolve scope issues in list comprehensions
BASE_DIR = os.environ.get("FONT_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))

class Config:
    STUDENT_ID = "2575736"
    
    SOURCE_FONT = f"{BASE_DIR}/source.ttf"
    TARGET_FONTS = [f"{BASE_DIR}/font{i}.ttf" for i in range(5)]
    
    # Dynamically read the poem file to ensure required characters are prioritized
    POEM_FILE = f"{BASE_DIR}/ExampleTangPoem.txt"
    
    # --- Organized Sampling Configuration ---
    SAMPLES_PER_FONT = 2000     # Target sample count per font
    CJK_START = 0x4E00         # CJK Unicode start
    CJK_END = 0x9FA5           # CJK Unicode end

def get_target_poem_chars():
    """Dynamically read and extract required characters from the test set (Tang Poem)."""
    if not os.path.exists(Config.POEM_FILE):
        print(f"[!] Warning: {Config.POEM_FILE} not found. Using Unicode sampling only.")
        return []
    with open(Config.POEM_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    return list(set([c for c in content if not c.isspace()]))

def render_char(char, font_path, img_size=256):
    """Render character and apply strict charset filtering using Pixel-level Cross-validation."""
    img = Image.new('L', (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, int(img_size * 0.8))
        
        # 1. Render the target character
        bbox = draw.textbbox((0, 0), char, font=font)
        if bbox is None: return None
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((img_size-w)/2, (img_size-h)/2 - bbox[1]), char, font=font, fill=0)
        img_array = np.array(img)
        
        # 2. Render a DEFINITIVELY MISSING character to get the [X] placeholder image
        # chr(0xFFFF) is an official Unicode Noncharacter, guaranteed to render as the missing glyph (.notdef)
        img_missing = Image.new('L', (img_size, img_size), color=255)
        draw_m = ImageDraw.Draw(img_missing)
        bbox_m = draw_m.textbbox((0, 0), chr(0xFFFF), font=font)
        if bbox_m is not None:
            w_m, h_m = bbox_m[2] - bbox_m[0], bbox_m[3] - bbox_m[1]
            draw_m.text(((img_size-w_m)/2, (img_size-h_m)/2 - bbox_m[1]), chr(0xFFFF), font=font, fill=0)
        missing_array = np.array(img_missing)

        # 3. Filter 1: Strict Pixel-level Equality Check
        # If the rendered character looks EXACTLY like the missing [X] box, discard it!
        if np.array_equal(img_array, missing_array):
            return None
        
        # 4. Filter 2: Blank or mostly blank images (99% threshold)
        if np.sum(img_array == 255) / (img_size * img_size) > 0.99:
            return None
            
        return img
    except Exception:
        return None

def main():
    print("[*] Starting automated data preprocessing pipeline...")
    
    # 1. Get required test set characters
    poem_chars = get_target_poem_chars()
    print(f"[*] Extracted {len(poem_chars)} required characters from the test set.")
    
    packaged_data = []

    # 2. Iterate through each target font
    for label, target_font in enumerate(Config.TARGET_FONTS):
        if not os.path.exists(target_font):
            print(f"[!] Skipping missing font: {target_font}")
            continue
            
        print(f"\n[*] Performing Organized Sampling for Font {label}...")
        valid_count = 0
        
        # Phase A: Prioritize sampling test set characters
        for char in poem_chars:
            src_img = render_char(char, Config.SOURCE_FONT)
            tgt_img = render_char(char, target_font)
            
            if src_img is not None and tgt_img is not None:
                packaged_data.append({'label': label, 'char': char, 'source_img': src_img, 'target_img': tgt_img})
                valid_count += 1
                
        print(f"    -> Sampled {valid_count} required test characters.")
        
        # Phase B: Systematically sample Unicode range until reaching SAMPLES_PER_FONT
        unicode_ptr = Config.CJK_START
        while valid_count < Config.SAMPLES_PER_FONT and unicode_ptr <= Config.CJK_END:
            char = chr(unicode_ptr)
            unicode_ptr += 1
            
            # Skip already sampled test characters
            if char in poem_chars:
                continue
                
            src_img = render_char(char, Config.SOURCE_FONT)
            tgt_img = render_char(char, target_font)
            
            # Strict charset filtering
            if src_img is not None and tgt_img is not None:
                packaged_data.append({'label': label, 'char': char, 'source_img': src_img, 'target_img': tgt_img})
                valid_count += 1
                
        print(f"    -> Font {label} sampling complete. Total valid samples: {valid_count}.")

    # 3. Package and save
    output_path = f"{Config.STUDENT_ID}-FontData.obj"
    with open(output_path, 'wb') as f:
        pickle.dump(packaged_data, f)
        
    print(f"\n[+] Preprocessing complete. Total packaged samples: {len(packaged_data)}")
    print(f"[+] Dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
