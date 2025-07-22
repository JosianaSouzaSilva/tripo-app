import os
import subprocess
from huggingface_hub import hf_hub_download, snapshot_download

os.makedirs("checkpoints", exist_ok=True)

if not os.path.exists("checkpoints/RealESRGAN_x2plus.pth"):
    print("[INIT] Baixando RealESRGAN...")
    hf_hub_download("dtarnow/UPscaler", filename="RealESRGAN_x2plus.pth", local_dir="checkpoints")

if not os.path.exists("checkpoints/big-lama.pt"):
    print("[INIT] Baixando Big-Lama...")
    subprocess.run("wget -P checkpoints/ https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt", shell=True, check=True)

if not os.path.exists("checkpoints/RMBG-1.4"):
    print("[INIT] Baixando RMBG-1.4...")
    snapshot_download('briaai/RMBG-1.4', local_dir='checkpoints/RMBG-1.4')

if not os.path.exists("checkpoints/TripoSG"):
    print("[INIT] Baixando TripoSG...")
    snapshot_download('VAST-AI/TripoSG', local_dir='checkpoints/TripoSG')

TRIPOSG_CODE_DIR = os.path.expandvars("$HOME/tripoSG")
if not os.path.exists(TRIPOSG_CODE_DIR):
    os.system(f"git clone https://github.com/VAST-AI-Research/TripoSG.git {TRIPOSG_CODE_DIR}")

MV_ADAPTER_CODE_DIR = os.path.expandvars("$HOME/mv_adapter")
if not os.path.exists(MV_ADAPTER_CODE_DIR):
    os.system(f"git clone https://github.com/huanngzh/MV-Adapter.git {MV_ADAPTER_CODE_DIR} && cd {MV_ADAPTER_CODE_DIR} && git checkout 7d37a97e9bc223cdb8fd26a76bd8dd46504c7c3d")
