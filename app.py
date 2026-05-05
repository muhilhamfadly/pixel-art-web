from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
from PIL import Image
import numpy as np
import io
import base64
import os
import uuid
import gc

app = Flask(__name__)
app.secret_key = 'pixel-art-converter-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024  

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

processed_images = {}

CONFIG = {
    'k_values': [8, 16, 24]
}

PIXEL_SIZES = [8, 16, 24, 32, 48]

PROCESS_DIMENSION = 512   
PREVIEW_DIMENSION = 1024   

def resize_to_dimension(image, max_dim):
    w, h = image.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def downsample(image, scale=10):
    new_w = max(1, image.width // scale)
    new_h = max(1, image.height // scale)
    return image.resize((new_w, new_h), Image.NEAREST)


def median_cut_optimized(pixels, k):
    n_pixels = len(pixels)

    if n_pixels == 0:
        return np.array([[128, 128, 128]], dtype=np.uint8)

    max_samples = 10000
    if n_pixels > max_samples:
        indices = np.random.choice(n_pixels, max_samples, replace=False)
        sample_pixels = pixels[indices]
    else:
        sample_pixels = pixels

    boxes = [sample_pixels.copy()]

    while len(boxes) < k:
        max_range = -1
        max_idx = 0
        for i, box in enumerate(boxes):
            if len(box) > 1:
                box_range = np.ptp(box, axis=0).max()
                if box_range > max_range:
                    max_range = box_range
                    max_idx = i

        box = boxes.pop(max_idx)

        if len(box) <= 1:
            boxes.append(box)
            continue

        ranges = np.ptp(box, axis=0)
        channel = np.argmax(ranges)

        median_idx = len(box) // 2
        indices = np.argpartition(box[:, channel], median_idx)

        boxes.append(box[indices[:median_idx]])
        boxes.append(box[indices[median_idx:]])

    palette = np.zeros((len(boxes), 3), dtype=np.float32)
    for i, box in enumerate(boxes):
        if len(box) > 0:
            palette[i] = np.mean(box, axis=0)
        else:
            palette[i] = [128, 128, 128]

    return palette.astype(np.uint8)


def quantize_image_optimized(image, k):
    img_np = np.array(image, dtype=np.uint8)
    h, w, _ = img_np.shape

    pixels = img_np.reshape(-1, 3).astype(np.float32)
    palette = median_cut_optimized(pixels, k).astype(np.float32)

    n_pixels = len(pixels)
    batch_size = 10000
    nearest = np.zeros(n_pixels, dtype=np.int32)

    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        batch = pixels[start:end]
        diff = batch[:, np.newaxis, :] - palette[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        nearest[start:end] = np.argmin(distances, axis=1)
        del diff, distances

    new_pixels = palette[nearest].reshape(h, w, 3)
    result = Image.fromarray(new_pixels.astype(np.uint8))
    palette_uint8 = palette.astype(np.uint8)

    del pixels, nearest, new_pixels, img_np
    gc.collect()

    return result, palette_uint8


def apply_palette_to_image(image_target, palette):
    target_np = np.array(image_target, dtype=np.uint8)
    h, w, _ = target_np.shape
    pixels = target_np.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)

    n_pixels = len(pixels)
    batch_size = 10000
    nearest = np.zeros(n_pixels, dtype=np.int32)

    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        batch = pixels[start:end]
        diff = batch[:, np.newaxis, :] - palette_f[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=2)
        nearest[start:end] = np.argmin(distances, axis=1)
        del diff, distances

    new_pixels = palette_f[nearest].reshape(h, w, 3)
    result = Image.fromarray(new_pixels.astype(np.uint8))

    del pixels, nearest, new_pixels, target_np
    gc.collect()

    return result


def upsample_pixel_art(image, target_size):
    return image.resize(target_size, Image.NEAREST)


def image_to_base64(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG', optimize=True)
    img_bytes.seek(0)
    return f"data:image/png;base64,{base64.b64encode(img_bytes.getvalue()).decode()}"


def process_variant(data, pixel_size, k, for_download=False):
    image_small    = data['image_small']     
    image_preview  = data['image_preview']    
    image_original = data['image_original']  

    downsampled_small = downsample(image_small, scale=pixel_size)
    _, palette = quantize_image_optimized(downsampled_small, k)

    image_target = image_original if for_download else image_preview

    downsampled_target = downsample(image_target, scale=pixel_size)
    quantized_target   = apply_palette_to_image(downsampled_target, palette)
    pixel_art          = upsample_pixel_art(quantized_target, image_target.size)

    del downsampled_small, palette, downsampled_target, quantized_target
    gc.collect()

    return pixel_art

# Flask Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        conversion_id = str(uuid.uuid4())
        image = Image.open(file.stream)

        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        image_original = image.copy()

        image_preview = resize_to_dimension(image.copy(), PREVIEW_DIMENSION)
        if image_preview.mode != 'RGB':
            image_preview = image_preview.convert('RGB')

        image_small = resize_to_dimension(image.copy(), PROCESS_DIMENSION)
        if image_small.mode != 'RGB':
            image_small = image_small.convert('RGB')

        del image  
        gc.collect()

        original_base64 = image_to_base64(image_preview)
        orig_w, orig_h = image_original.size

        processed_images[conversion_id] = {
            'image_original': image_original,  
            'image_preview':  image_preview,   
            'image_small':    image_small,      
            'original':       original_base64,
            'cache_preview':  {},               
            'filename':       file.filename,
            'k_values':       CONFIG['k_values'],
            'pixel_sizes':    PIXEL_SIZES,
            'original_width': orig_w,
            'original_height': orig_h
        }

        MAX_CACHE = 2
        if len(processed_images) > MAX_CACHE:
            oldest_key = next(iter(processed_images))
            del processed_images[oldest_key]

        gc.collect()
        return jsonify({'success': True, 'conversion_id': conversion_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_variant/<conversion_id>/<int:pixel_size>/<int:k>')
def get_variant(conversion_id, pixel_size, k):
    if conversion_id not in processed_images:
        return jsonify({'error': 'Not found'}), 404

    data = processed_images[conversion_id]
    cache_key = f"{pixel_size}_{k}"

    if cache_key not in data['cache_preview']:
        pixel_art = process_variant(data, pixel_size, k, for_download=False)
        result = image_to_base64(pixel_art)

        # Batasi cache preview (maks 9 kombinasi)
        if len(data['cache_preview']) >= 9:
            oldest = next(iter(data['cache_preview']))
            del data['cache_preview'][oldest]

        data['cache_preview'][cache_key] = result
        gc.collect()

    return jsonify({'image': data['cache_preview'][cache_key]})


@app.route('/loading/<conversion_id>')
def loading(conversion_id):
    return render_template('loading.html', conversion_id=conversion_id)


@app.route('/result/<conversion_id>')
def result(conversion_id):
    if conversion_id not in processed_images:
        return redirect(url_for('index'))

    data = processed_images[conversion_id]

    return render_template('result.html',
                           conversion_id=conversion_id,
                           original=data['original'],
                           filename=data['filename'],
                           k_values=data['k_values'],
                           pixel_sizes=data['pixel_sizes'],
                           original_width=data['original_width'],
                           original_height=data['original_height'])


@app.route('/check/<conversion_id>')
def check_conversion(conversion_id):
    if conversion_id in processed_images:
        return jsonify({'ready': True})
    return jsonify({'ready': False})


@app.route('/download/<conversion_id>/<pixel_size>/<k_value>')
def download(conversion_id, pixel_size, k_value):
    if conversion_id not in processed_images:
        return jsonify({'error': 'Not found'}), 404

    try:
        ps = int(pixel_size)
        k  = int(k_value)
    except ValueError:
        return jsonify({'error': 'Invalid parameters'}), 400

    data = processed_images[conversion_id]

    # Proses dengan resolusi asli
    pixel_art = process_variant(data, ps, k, for_download=True)

    img_bytes = io.BytesIO()
    pixel_art.save(img_bytes, format='PNG', optimize=True)
    img_bytes.seek(0)

    del pixel_art
    gc.collect()

    orig_w = data['original_width']
    orig_h = data['original_height']
    filename = f"pixel_art_{pixel_size}px_{k_value}colors_{orig_w}x{orig_h}.png"

    return send_file(
        img_bytes,
        mimetype='image/png',
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)