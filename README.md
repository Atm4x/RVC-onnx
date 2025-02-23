# ONNX RVC test

## Installation

### 1) Cloning
```bash
git clone https://github.com/Atm4x/RVC-onnx.git
cd RVC-onnx
```

### 2) Installing UV
```bash
pip install uv -U
uv --version
```

### 3) Setup VENV

```bash
uv venv --python 3.10
.venv\Scripts\activate
```

### 4) Install all deps
```bash
uv pip install -r requirements.txt
```

## Usage

#### 1) Download .onnx model and place it into `onnx_folder` folder.
#### 2) Put your .wav audio file into `input` folder.
#### 3) Run the program and wait for the result:
```bash
uv run RVC_Onnx_Infer.py
```



