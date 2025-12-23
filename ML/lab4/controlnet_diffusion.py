import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore', message='.*NumPy.*')

import torch
import numpy as np

if not hasattr(torch, 'xpu'):
    class DeviceCountCallable:
        def __call__(self):
            return 0
        def __int__(self):
            return 0
        def __index__(self):
            return 0
        def __repr__(self):
            return '0'
    
    class XPUStub:
        device_count = DeviceCountCallable()
        
        @staticmethod
        def empty_cache():
            pass
        
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def get_device_name(device=None):
            return "XPU"
        
        @staticmethod
        def current_device():
            return None
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    torch.xpu = XPUStub()

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image


def create_canny_image(image_path, low_threshold=100, high_threshold=200):
    import cv2
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, low_threshold, high_threshold)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    canny_image = Image.fromarray(canny)
    return canny_image


def load_controlnet_pipeline(device='cuda', model_id="CompVis/stable-diffusion-v1-4", 
                             controlnet_model_id="lllyasviel/sd-controlnet-canny"):
    cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    dtype = torch.float16 if device == 'cuda' else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=cache_dir
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    if device == 'cuda':
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    
    return pipe


def generate_with_controlnet(pipe, control_image, prompt, negative_prompt="", 
                            num_inference_steps=20, guidance_scale=7.5, 
                            num_images=1, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    images = pipe(
        prompt,
        image=control_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images
    
    return images


def experiment_with_parameters(pipe, control_image, prompt, negative_prompt="",
                              output_dir="outputs"):
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = [
        {
            "name": "baseline",
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": 42
        },
        {
            "name": "high_guidance",
            "num_inference_steps": 20,
            "guidance_scale": 15.0,
            "seed": 42
        },
        {
            "name": "low_guidance",
            "num_inference_steps": 20,
            "guidance_scale": 3.0,
            "seed": 42
        },
        {
            "name": "more_steps",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "seed": 42
        },
        {
            "name": "fewer_steps",
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
            "seed": 42
        },
        {
            "name": "different_seed",
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "seed": 123
        },
    ]
    
    results = []
    
    for exp in tqdm(experiments, desc="Генерация изображений"):
        images = generate_with_controlnet(
            pipe=pipe,
            control_image=control_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=exp['num_inference_steps'],
            guidance_scale=exp['guidance_scale'],
            seed=exp['seed']
        )
        
        image = images[0]
        filename = f"{exp['name']}_steps{exp['num_inference_steps']}_guidance{exp['guidance_scale']:.1f}_seed{exp['seed']}.png"
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        
        results.append({
            'name': exp['name'],
            'image': image,
            'params': exp,
            'filepath': filepath
        })
    
    create_comparison_grid(results, control_image, output_dir, prompt)
    
    return results


def create_comparison_grid(results, control_image, output_dir, prompt):
    n_results = len(results)
    cols = 3
    rows = (n_results + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    axes[0].imshow(control_image)
    axes[0].set_title("Control Image (Canny)", fontsize=10)
    axes[0].axis('off')
    
    for idx, result in enumerate(results, start=1):
        axes[idx].imshow(result['image'])
        title = (f"{result['name']}\n"
                f"Steps: {result['params']['num_inference_steps']}, "
                f"Guidance: {result['params']['guidance_scale']:.1f}")
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis('off')
    
    for idx in range(len(results) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"ControlNet Experiments\nPrompt: {prompt[:60]}...", 
                 fontsize=12, y=0.995)
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, "comparison_grid.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_simple_diffusion(prompt, negative_prompt="", num_inference_steps=20,
                             guidance_scale=7.5, num_images=1, seed=None,
                             model_id="CompVis/stable-diffusion-v1-4",
                             device='cuda', output_dir="outputs"):
    from diffusers import StableDiffusionPipeline
    
    dtype = torch.float16 if device == 'cuda' else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    if device == 'cuda':
        pipe = pipe.to(device)
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, image in enumerate(images):
        filename = f"simple_diffusion_{idx}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
    
    return images


def main():
    parser = argparse.ArgumentParser(description='ControlNet и диффузия эксперименты')
    parser.add_argument('--mode', type=str, default='controlnet',
                       choices=['controlnet', 'simple', 'both'],
                       help='Режим работы: controlnet, simple или both')
    parser.add_argument('--control_image', type=str, default=None,
                       help='Путь к контрольному изображению для ControlNet')
    parser.add_argument('--prompt', type=str, 
                       default='a beautiful landscape, highly detailed, 4k',
                       help='Промпт для генерации')
    parser.add_argument('--negative_prompt', type=str, default='',
                       help='Негативный промпт')
    parser.add_argument('--num_inference_steps', type=int, default=20,
                       help='Количество шагов инференса')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Guidance scale (CFG)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Seed для воспроизводимости')
    parser.add_argument('--num_images', type=int, default=1,
                       help='Количество изображений для генерации')
    parser.add_argument('--experiment', action='store_true',
                       help='Запустить эксперименты с разными параметрами')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Директория для сохранения результатов')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство (cuda/cpu), по умолчанию автоопределение')
    parser.add_argument('--canny_low', type=int, default=100,
                       help='Нижний порог для Canny edge detection')
    parser.add_argument('--canny_high', type=int, default=200,
                       help='Верхний порог для Canny edge detection')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.exists('/app/outputs'):
        base_output_dir = '/app/outputs'
    elif not os.path.isabs(args.output_dir):
        base_output_dir = os.path.join(os.getcwd(), args.output_dir)
    else:
        base_output_dir = args.output_dir
    
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.mode in ['controlnet', 'both']:
        if args.control_image is None:
            test_image = Image.new('RGB', (512, 512), color='white')
            import cv2
            import numpy as np
            img_array = np.array(test_image)
            cv2.rectangle(img_array, (100, 100), (400, 400), (0, 0, 0), 5)
            cv2.circle(img_array, (250, 250), 50, (0, 0, 0), 3)
            test_image = Image.fromarray(img_array)
            control_image_path = os.path.join(output_dir, "test_control.png")
            test_image.save(control_image_path)
            args.control_image = control_image_path
        
        control_image = create_canny_image(
            args.control_image, 
            low_threshold=args.canny_low,
            high_threshold=args.canny_high
        )
        canny_path = os.path.join(output_dir, "canny_control.png")
        control_image.save(canny_path)
        
        pipe = load_controlnet_pipeline(device=device)
        
        if args.experiment:
            experiment_with_parameters(
                pipe=pipe,
                control_image=control_image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                output_dir=output_dir
            )
        else:
            images = generate_with_controlnet(
                pipe=pipe,
                control_image=control_image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images=args.num_images,
                seed=args.seed
            )
            
            for idx, image in enumerate(images):
                filename = f"controlnet_result_{idx}.png"
                if not os.path.isabs(output_dir):
                    output_dir = os.path.abspath(output_dir)
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
    
    if args.mode in ['simple', 'both']:
        generate_simple_diffusion(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images=args.num_images,
            seed=args.seed,
            device=device,
            output_dir=output_dir
        )
    
    print(f"Результаты сохранены в: {output_dir}")


if __name__ == '__main__':
    main()
