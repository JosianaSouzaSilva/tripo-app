import os
import sys
import json
import boto3, botocore
import torch
import numpy as np
from PIL import Image
import trimesh
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
import subprocess
import xatlas
from setup_pipeline import TRIPOSG_CODE_DIR, MV_ADAPTER_CODE_DIR

# Adiciona os repositórios clonados ao sys.path
sys.path.append(TRIPOSG_CODE_DIR)
sys.path.append(os.path.join(TRIPOSG_CODE_DIR, "scripts"))
sys.path.append(MV_ADAPTER_CODE_DIR)
sys.path.append(os.path.join(MV_ADAPTER_CODE_DIR, "scripts"))

from image_process import prepare_image
from briarmbg import BriaRMBG
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from inference_ig2mv_sdxl import prepare_pipeline, preprocess_image, remove_bg
from mvadapter.utils import get_orthogonal_camera, make_image_grid
from mvadapter.utils.render import NVDiffRastContextWrapper, load_mesh, render
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Usando dispositivo: {DEVICE}")
if DEVICE != "cuda":
    raise RuntimeError("CUDA não está disponível. Verifique se o PyTorch está instalado corretamente com suporte a CUDA.")

DTYPE = torch.float16
NUM_VIEWS = 6
DEFAULT_FACE_NUMBER = 10000
TMP_DIR = "/tmp/ec2_tripo_job"

if os.path.exists(TMP_DIR):
    print(f"[DIR] Limpando o diretório temporário: {TMP_DIR}")
    shutil.rmtree(TMP_DIR, ignore_errors=True)

os.makedirs(TMP_DIR, exist_ok=True)

INPUT_BUCKET = os.environ["INPUT_BUCKET"]
OUTPUT_BUCKET = os.environ["OUTPUT_BUCKET"]
if not INPUT_BUCKET or not OUTPUT_BUCKET:
    raise ValueError("As variáveis de ambiente INPUT_BUCKET e OUTPUT_BUCKET devem ser definidas.")

s3 = boto3.client("s3")

def get_random_hex():
    return os.urandom(8).hex()

def generate_temp_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    temp_name = f"{name}_{get_random_hex()}{ext}"
    print(f"[generate_temp_filename] Base: {filename} -> Temp: {temp_name}")
    return temp_name
    
def download_from_s3(bucket, key, local_path):
    print(f"[S3] Download: s3://{bucket}/{key} -> {local_path}")
    try:
        s3.download_file(bucket, key, local_path)
    except botocore.exceptions.BotoCoreError as e:
        raise RuntimeError(f"Erro ao baixar {key} do bucket {bucket}: {e}")
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Download de {key} concluído, mas o arquivo não foi encontrado em {local_path}")
    print("[S3] Download concluído com sucesso")

def upload_to_s3(local_path, bucket, key):
    print(f"[S3] Upload: {local_path} -> s3://{bucket}/{key}")
    try:
        s3.upload_file(local_path, bucket, key)
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    except botocore.exceptions.BotoCoreError as e:
        raise RuntimeError(f"Erro ao fazer upload de {local_path} para o bucket {bucket}: {e}")
    print("[S3] Upload concluído com sucesso")

def ensure_uv_mapping(mesh_path):
    """
    Garante que o mesh tenha UV. Se não tiver, faz o unwrap com xatlas e salva novamente.
    """
    print("[UV] Verificando UV no mesh...")
    mesh = trimesh.load(mesh_path)
    mesh_uv = None
    if isinstance(mesh, trimesh.Scene):
        mesh_list = [g for g in mesh.geometry.values()]
        mesh_uv = mesh_list[0] if mesh_list else None
    else:
        mesh_uv = mesh

    if mesh_uv is not None and hasattr(mesh_uv.visual, "uv") and mesh_uv.visual.uv is not None:
        print("[UV] Mesh já tem UV.")
        return

    print("[UV] Mesh NÃO tem UV. Aplicando unwrap com xatlas...")
    try:
        v = mesh_uv.vertices.astype(np.float32)
        f = mesh_uv.faces.astype(np.uint32)
        
        atlas = xatlas.Atlas()
        atlas.add_mesh(v, f)
        chart_options = xatlas.ChartOptions()
        pack_options = xatlas.PackOptions()
        
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        vmapping, indices, uvs = atlas.get_mesh(0)
        
        mesh_uv.vertices = mesh_uv.vertices[vmapping]
        mesh_uv.faces = indices
        mesh_uv.visual = trimesh.visual.TextureVisuals(uv=uvs)
        
        mesh_uv.export(mesh_path)
        print("[UV] UV mapping gerado e mesh salvo!")
    except Exception as e:
        print(f"[UV][ERRO] Falha ao aplicar UV mapping: {type(e).__name__}: {e}")
        raise

def main(job_id, image_name="input.png", *args, **kwargs):
    seg_url = mesh_url = textured_url = mv_url = None

    try:
        print(f"\n=== Iniciando Job {job_id} ===")
        input_prefix = output_prefix = f"{job_id}/"
        input_img_key = input_prefix + image_name
        local_input_img = os.path.join(TMP_DIR, image_name)
        download_from_s3(INPUT_BUCKET, input_img_key, local_input_img)

        print("[MODEL] Carregando RMBG e pipelines...")
        try:
            rmbg_net = BriaRMBG.from_pretrained("checkpoints/RMBG-1.4").to(DEVICE).eval()
            triposg_pipe = TripoSGPipeline.from_pretrained("checkpoints/TripoSG").to(DEVICE, DTYPE)
            mv_adapter_pipe = prepare_pipeline(
                base_model="stabilityai/stable-diffusion-xl-base-1.0",
                vae_model="madebyollin/sdxl-vae-fp16-fix",
                unet_model=None,
                lora_model=None,
                adapter_path="huanngzh/mv-adapter",
                scheduler=None,
                num_views=NUM_VIEWS,
                device=DEVICE,
                dtype=torch.float16
            )
        except Exception as e:
            print(f"[ERRO][MODEL] Falha ao carregar pipelines/modelos: {type(e).__name__}: {e}")
            raise

        try:
            birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(DEVICE)
        except Exception as e:
            print(f"[ERRO][MODEL] Falha ao carregar BiRefNet: {type(e).__name__}: {e}")
            raise

        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        def remove_bg_fn(img):
            try:
                # Garante que a imagem seja RGB antes de processar
                if isinstance(img, Image.Image):
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                elif isinstance(img, np.ndarray):
                    if len(img.shape) == 2:
                        img = np.stack([img]*3, axis=-1)
                    elif img.shape[2] == 4:
                        img = img[:, :, :3]
                return remove_bg(img, birefnet, transform_image, DEVICE)
            except Exception as e:
                print(f"[ERRO][REMOVE_BG] {type(e).__name__}: {e}")
                raise

        seed = 0
        num_inference_steps = 50
        guidance_scale = 7.5
        simplify = True
        target_face_num = DEFAULT_FACE_NUMBER
        text_prompt = "high quality"

        print("[STEP] Segmentação...")
        try:
            image_seg = prepare_image(local_input_img, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
            seg_path = os.path.join(TMP_DIR, generate_temp_filename("segmentation.png"))
            if isinstance(image_seg, np.ndarray):
                Image.fromarray((image_seg * 255).astype(np.uint8)).save(seg_path)
            else:
                image_seg.save(seg_path)
            print(f"[STEP] Segmentação salva em {seg_path}")
            seg_url = upload_to_s3(seg_path, OUTPUT_BUCKET, output_prefix + "segmentation.png")
        except Exception as e:
            print(f"[ERRO][SEGMENTAÇÃO] {type(e).__name__}: {e}")
            raise

        print("[STEP] Reconstrução Mesh 3D...")
        try:
            outputs = triposg_pipe(
                image=image_seg,
                generator=torch.Generator(device=triposg_pipe.device).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).samples[0]
            mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
            if simplify:
                print("[STEP] Simplificando mesh...")
                from utils import simplify_mesh
                mesh = simplify_mesh(mesh, target_face_num)
            mesh_path = os.path.join(TMP_DIR, generate_temp_filename("triposg.glb"))
            mesh.export(mesh_path)
            print(f"[STEP] Mesh exportado para {mesh_path}")
        except Exception as e:
            print(f"[ERRO][RECONSTRUÇÃO MESH] {type(e).__name__}: {e}")
            raise

        ensure_uv_mapping(mesh_path)

        print("[DEBUG] Verificando se o mesh tem UV:")
        try:
            loaded_mesh = trimesh.load(mesh_path)
            mesh_uv = None
            if isinstance(loaded_mesh, trimesh.Scene):
                mesh_list = [g for g in loaded_mesh.geometry.values()]
                mesh_uv = mesh_list[0] if mesh_list else None
            else:
                mesh_uv = loaded_mesh

            if mesh_uv is not None and hasattr(mesh_uv.visual, "uv") and mesh_uv.visual.uv is not None:
                print(f"[DEBUG] UV encontrado: {mesh_uv.visual.uv.shape}")
            else:
                print("[ALERTA] Mesh exportado NÃO TEM UV!")
        except Exception as e:
            print(f"[ERRO][CHECK UV] {type(e).__name__}: {e}")

        try:
            mesh_url = upload_to_s3(mesh_path, OUTPUT_BUCKET, output_prefix + "model.glb")
        except Exception as e:
            print(f"[ERRO][UPLOAD MESH] {type(e).__name__}: {e}")

        height, width = 768, 768
        cameras = get_orthogonal_camera(
            elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
            distance=[1.8] * NUM_VIEWS,
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            device=DEVICE,
        )
        ctx = NVDiffRastContextWrapper(device=DEVICE, context_type="cuda")
        mesh_loaded = load_mesh(mesh_path, rescale=True, device=DEVICE)

        try:
            print("[STEP] Renderizando views...")
            render_out = render(ctx, mesh_loaded, cameras, height=height, width=width, render_attr=False, normal_background=0.0)
            control_images = torch.cat([
                (render_out.pos[..., :3] + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ], dim=-1).permute(0, 3, 1, 2).to(DEVICE)
        except Exception as e:
            print(f"[ERRO][RENDER] {type(e).__name__}: {e}")
            raise

        try:
            print("[STEP] Preprocessando imagem para referência (remoção de BG e RGB)...")
            image = Image.open(local_input_img)
            
            # Garante que a imagem seja RGB
            if image.mode == 'RGBA':
                # Cria fundo branco para imagens com transparência
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Remove background
            image = remove_bg_fn(image)
            
            # Garante que a saída seja RGB
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = np.stack([image]*3, axis=-1)
                elif image.shape[2] > 3:
                    image = image[:, :, :3]
                image = Image.fromarray((image * 255).astype(np.uint8))
            
            print(f"[DEBUG] Após remoção de BG, modo: {image.mode}, size: {image.size}")
            
            # Pré-processamento final
            image = preprocess_image(image, height, width)
            
            # Garante que a imagem tenha 3 canais
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                elif image.shape[0] > 3:
                    image = image[:3, :, :]
        except Exception as e:
            print(f"[ERRO][PREPROCESS IMAGEM TEX] {type(e).__name__}: {e}")
            raise

        try:
            print("[STEP] Gerando views MV-Adapter...")
            images = mv_adapter_pipe(
                text_prompt, height=height, width=width,
                num_inference_steps=10, guidance_scale=3.0, num_images_per_prompt=NUM_VIEWS,
                control_image=control_images, control_conditioning_scale=1.0,
                reference_image=image, reference_conditioning_scale=1.0,
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                cross_attention_kwargs={"scale": 1.0},
                generator=torch.Generator(device=DEVICE).manual_seed(seed)
            ).images

            mv_image_path = os.path.join(TMP_DIR, generate_temp_filename("mv_adapter.png"))
            make_image_grid(images, rows=1).save(mv_image_path)
            print(f"[STEP] Views salvos em {mv_image_path}")

            # Garante que a imagem de textura seja RGB
            img = Image.open(mv_image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
                img.save(mv_image_path)
            img.close()
        except Exception as e:
            print(f"[ERRO][MV-ADAPTER/TEX VIEWS] {type(e).__name__}: {e}")
            raise

        try:
            print("[STEP] Gerando textura com TexturePipeline...")
            from texture import TexturePipeline, ModProcessConfig

            texture_pipe = TexturePipeline(
                upscaler_ckpt_path="checkpoints/RealESRGAN_x2plus.pth",
                inpaint_ckpt_path="checkpoints/big-lama.pt",
                device=DEVICE,
            )
            textured_glb_path = texture_pipe(
                mesh_path=mesh_path,
                save_dir=TMP_DIR,
                save_name=generate_temp_filename("texture_mesh.glb"),
                uv_unwarp=True, uv_size=4096, rgb_path=mv_image_path,
                rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
                camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            )
            print(f"[STEP] Mesh texturizado salvo em {textured_glb_path}")

            # Verificação de textura aplicada
            try:
                loaded_mesh_tex = trimesh.load(textured_glb_path)
                mesh_tex_check = None
                if isinstance(loaded_mesh_tex, trimesh.Scene):
                    mesh_list = [g for g in loaded_mesh_tex.geometry.values()]
                    mesh_tex_check = mesh_list[0] if mesh_list else None
                else:
                    mesh_tex_check = loaded_mesh_tex

                if mesh_tex_check is not None and hasattr(mesh_tex_check.visual, "material") and mesh_tex_check.visual.material.image is not None:
                    print("[DEBUG] Textura aplicada no mesh final!")
                else:
                    print("[ALERTA] Mesh final NÃO TEM textura aplicada!")
            except Exception as ex:
                print(f"[ERRO][CHECK MESH TEX] {type(ex).__name__}: {ex}")

            textured_url = upload_to_s3(textured_glb_path, OUTPUT_BUCKET, output_prefix + "textured.glb")
            mv_url = upload_to_s3(mv_image_path, OUTPUT_BUCKET, output_prefix + "views.png")
        except Exception as e:
            print(f"[ERRO][TEXTURE PIPELINE] {type(e).__name__}: {e}")

        result = {"segmentation": seg_url, "model": mesh_url, "textured": textured_url, "views": mv_url}
        result_path = os.path.join(TMP_DIR, "result.json")
        try:
            with open(result_path, "w") as f:
                json.dump(result, f)
            print(f"[STEP] Salvando resultado final em {result_path}")
            upload_to_s3(result_path, OUTPUT_BUCKET, output_prefix + "result.json")
        except Exception as e:
            print(f"[ERRO][SALVAR RESULTADO] {type(e).__name__}: {e}")

        print("Processamento finalizado com sucesso!")
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    except Exception as e:
        print(f"[ERRO][PIPELINE] {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py <job_id> [image_name]")
        sys.exit(1)
    job_id = sys.argv[1]
    image_name = sys.argv[2] if len(sys.argv) > 2 else "input.png"
    main(job_id, image_name)