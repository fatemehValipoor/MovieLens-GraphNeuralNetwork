@echo off
REM Step 1: Create virtual environment
python -m venv .venv

REM Step 2: Activate virtual environment
call .venv\Scripts\activate

REM Step 3: Upgrade pip
pip install --upgrade pip

REM Step 4: Install core scientific libraries
pip install numpy pandas matplotlib seaborn scikit-learn tqdm networkx jupyterlab

REM Step 5: Install ipython/ipykernel for Jupyter support
pip install ipython ipykernel

REM Step 6: Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Step 7: Install PyTorch Geometric and dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-geometric

REM Step 8: Register as Jupyter kernel
python -m ipykernel install --user --name=gnn-gpu-env --display-name "Python (GNN GPU)"

echo.
echo ✅ GPU-compatible environment is ready and registered as Jupyter kernel: "Python (GNN GPU)"
pause
