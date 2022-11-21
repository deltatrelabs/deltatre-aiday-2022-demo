# Optimizing Deep Learning models: theory, tools & best-practices (AI Day 2022)

In this repository you can find the slides and demo for **Optimizing Deep Learning models: theory, tools & best-practices** session, presented (in Italian) at [AI Day 2022 Conference](https://aiday.dotnetdev.it/) on November 18th, 2022.

Abstract:

In this session we'll discuss the possible different layers of incremental improvements for deep learning models, from architecture review to dataset cleaning, passing by training pipelines optimization and helpful tools to use.

Speakers:

- [Clemente Giorio](https://www.linkedin.com/in/clemente-giorio-03a61811/) (Deltatre, Microsoft MVP)
- [Gianni Rosa Gallina](https://www.linkedin.com/in/gianni-rosa-gallina-b206a821/) (Deltatre, Microsoft MVP)

---

## Demo 1 - OpenVINO Demo

The notebook shows how to take an ONNX model, convert/optimize it to OpenVINO IR format, and run it in the OpenVINO runtime. The optimized model is compared against the original ONNX model, for output compatibility and performance evaluation.

Requirements: **Python 3.9.x**, **OpenVINO 2022.2**, **ONNX Runtime 1.13.1** (on Windows)  
NVIDIA GPU (with CUDA 11.6 and cuDNN)

Setup environment following the [official installation guide](https://github.com/openvinotoolkit/openvino_notebooks#-installation-guide) and the steps below to configure a Python Virtual Environment.  
For additional details and other examples, please refer to the [OpenVINO Notebooks repository](https://github.com/openvinotoolkit/openvino_notebooks). We used the [102-pytorch-onnx-to-openvino notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/102-pytorch-onnx-to-openvino) as a starting point for this demo.

```powershell
python -m venv .venv
.\.venv\scripts\activate
python -m pip install -U pip
pip install wheel
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install openvino-dev[onnx]==2022.2.0
pip install fastseg
pip install ipywidgets
pip install matplotlib
```

## Demo 2 - ONNX Runtime

A demo script shows how to take an ONNX model, optimize it (static quantization), and run it in the ONNX Runtime. The optimized model is compared against the original ONNX model, for output compatibility and performance evaluation.

Requirements: **Python 3.9.x**, **ONNX Runtime 1.13.1** (on Windows)  
NVIDIA GPU (with CUDA 11.6 and cuDNN)

For additional details and other examples, please refer to the [ONNX Runtime docs](https://onnxruntime.ai/docs/performance/quantization.html). We took inspiration from the [
ONNX Runtime Inference Examples repository](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization) for the demo script.

Follow the steps below to configure a Python Virtual Environment.

```powershell
python -m venv .venv
.\.venv\scripts\activate
python -m pip install -U pip
pip install wheel
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install onnxruntime_gpu
```

## Demo 3 - Nebuly

A demo script shows how to take a PyTorch model and optimize it leveraging the Nebuly's [Nebullvum](https://github.com/nebuly-ai/nebullvm) open-source library. The optimized model is compared against the original model, for output compatibility and performance evaluation on different frameworks, returning the best one for the available hardware.

Requirements: **Python 3.9.x**, **PyTorch 1.12.1**, **Nebullvm 0.5+**, **TensorRT 8.5.1.7**, **Intel OpenVINO 2022.2**, **ONNX Runtime 1.13.1** (on Linux/WSL Ubuntu 20.04)  
NVIDIA GPU (with CUDA 11.6, cuDNN)

Setup environment following the [official installation guide](https://nebuly.gitbook.io/nebuly/nebullvm/installation) and/or the steps below to configure a Python Virtual Environment.  
For additional details and other examples, please refer to the [Nebuly docs](https://nebuly.gitbook.io/nebuly/nebullvm/installation).

```bash
python3 -m venv .venv
source ./.venv/bin/activate
python3 -m pip install -U pip
pip install wheel
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# See: https://nebuly.gitbook.io/nebuly/nebullvm/installation#use-the-auto-installer-recommended
pip install git+https://github.com/nebuly-ai/nebullvm.git
python3 -m nebullvm.installers.auto_installer --frameworks torch onnx --compilers torch_tensor_rt intel_neural_compressor deepsparse openvino
# See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
python3 -m pip install --upgrade tensorrt 
```

## License
---

Copyright (C) 2022 Deltatre.  
Licensed under [CC BY-NC-SA 4.0](./LICENSE).
