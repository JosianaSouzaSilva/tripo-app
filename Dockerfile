FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/app

WORKDIR $HOME

# Instala dependências básicas e Python
RUN mkdir -p /var/lib/apt/lists/partial && \ 
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv python3-dev \
    git wget curl libgl1-mesa-glx libglib2.0-0 libxrender1 libsm6 libxext6 \
    ffmpeg libglfw3 unzip libx11-6 libxi6 libxrandr2 libxxf86vm1 libxinerama1 libxcursor1 \
    git-lfs && \
    git lfs install --skip-repo && \
    rm -rf /var/lib/apt/lists/*

# Alias para python3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Cria usuário e diretório
RUN useradd -m user && \
    mkdir -p $HOME && \
    chown -R user:user $HOME

USER user

# Copia arquivos -
COPY --chown=user:user . $HOME

# Instala as dependências do TripoSG e MV-Adapter
COPY --chown=user:user nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl .
COPY --chown=user:user diso-0.1.4-cp310-cp310-linux_x86_64.whl .

RUN pip install --no-cache-dir \
./nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl \
    ./diso-0.1.4-cp310-cp310-linux_x86_64.whl
    
COPY --chown=user:user texture.cpython-310-x86_64-linux-gnu.so /app/texture.cpython-310-x86_64-linux-gnu.so

# ------ Bibliotecas -----
# RUN git clone https://github.com/VAST-AI-Research/TripoSG.git $HOME/tripoSG
# ENV PYTHONPATH="${PYTHONPATH}:$HOME/tripoSG"

# RUN git clone https://github.com/huanngzh/MV-Adapter.git $HOME/mv_adapter && \
#     cd $HOME/mv_adapter && \
#     git checkout 7d37a97e9bc223cdb8fd26a76bd8dd46504c7c3d
# ENV PYTHONPATH="${PYTHONPATH}:$HOME/mv_adapter"

# ------- Modelos do Hugging Face ------
# Install huggingface_hub
RUN pip install huggingface_hub
    
# Download RealESRGAN_x2plus.pth from HuggingFace
#RUN python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('dtarnow/UPscaler', filename='RealESRGAN_x2plus.pth', local_dir='checkpoints')\"

# Download big-lama.pt using wget
#RUN wget -P checkpoints/ https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt

# Download RMBG-1.4 and TripoSG snapshots from HuggingFace
#RUN python -c \"from huggingface_hub import snapshot_download; snapshot_download('briaai/RMBG-1.4', local_dir='checkpoints/RMBG-1.4')\"
#RUN python -c \"from huggingface_hub import snapshot_download; snapshot_download('VAST-AI/TripoSG', local_dir='checkpoints/TripoSG')\"

# Verifica a instação das bibliotecas e modelos
RUN python setup_pipeline.py

# ------ Instala o restante das libs -----
# RUN pip install --no-cache-dir -r requirements.txt
ENV TMPDIR=/app/tmp
RUN mkdir -p /app/tmp

RUN pip install --no-cache-dir -r requirements-base.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instala PyTorch com CUDA 11.8
RUN pip install --trusted-host download.pytorch.org \
--index-url http://download.pytorch.org/whl/cu118 \
torch torchvision torchaudio

EXPOSE 8000

CMD ["python", "app.py"]