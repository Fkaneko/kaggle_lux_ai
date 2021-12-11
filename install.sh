conda install -y mamba -n base -c conda-forge

mamba install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia

# basic install
mamba install -y \
    numpy \
    pandas \
    matplotlib \
    scipy \
    seaborn \
    scikit-learn==1.0.1

mamba install -y -c plotly plotly=4.14.3
pip install \
    pytorch-lightning==1.4.9 \
    kaggle \
    albumentations

pip install hydra-core --upgrade


pip install \
    onnx \
    onnxruntime

cd LuxPythonEnvGym
python setup.py install
cd ../

cd HandyRL
pip install -r requirements.txt
cd ../
