<p align="center">
 <img src="./assets/lumina2-logo.png" width="40%"/>
 <br>
</p>

<div align="center">
<h1> Lumina-Image 2.0: A Unified and Efficient Image Generative Framework </h1>

</div>
<div align="center">

[![Lumina-Next](https://img.shields.io/badge/Paper-Lumina--Image--2.0-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2503.21758)&#160;
[![Badge](https://img.shields.io/badge/-WeChat@Join%20Our%20Group-000000?logo=wechat&logoColor=07C160)](http://47.100.29.251:11001/)&#160;

[![Static Badge](https://img.shields.io/badge/Lumina--Image--2.0%20checkpoints-Model(2B)-yellow?logoColor=violet&label=%F0%9F%A4%97%20Lumina-Image-2.0%20checkpoints)](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)
[![Static Badge](https://img.shields.io/badge/Lumina--Image--2.0-HF_Space-yellow?logoColor=violet&label=%F0%9F%A4%97%20Demo%20Lumina-Image-2.0)](https://huggingface.co/spaces/Alpha-VLLM/Lumina-Image-2.0)
[![Lumina-Next](https://img.shields.io/badge/Lumina--Image--2.0-Diffusers-yellow?logo=Lumina-Image-2.0&logoColor=yellow)]([https://arxiv.org/abs/2503.21758](https://huggingface.co/docs/diffusers/main/en/api/pipelines/lumina2))&#160;


[![Static Badge](https://img.shields.io/badge/Huiying-6B88E3?logo=youtubegaming&label=Demo%20Lumina-Image-2.0)](https://magic-animation.intern-ai.org.cn/image/create)&#160;
[![Static Badge](https://img.shields.io/badge/Gradio-6B88E3?logo=youtubegaming&label=Demo%20Lumina-Image-2.0)](http://47.100.29.251:10010/)&#160;

</div>







## 📰 News
- [2025-3-28] 👋👋👋 We are excited to announce the release of the Lumina-Image 2.0 [Tech Report](https://arxiv.org/abs/2503.21758). We welcome discussions and feedback! 
- [2025-2-20] Diffusers team released a LoRA fine-tuning script for Lumina2. Find out more [here](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_lumina2.md).
- [2025-2-12] Lumina 2.0 is now available in Diffusers. Check out the [docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/lumina2) to know more.
- [2025-2-10] The official [Hugging Face Space](https://huggingface.co/spaces/Alpha-VLLM/Lumina-Image-2.0) for Lumina-Image 2.0 is now available.
- [2025-2-10] Preliminary explorations of video generation with **[Lumina-Video 1.0](https://github.com/Alpha-VLLM/Lumina-Video)** have been released.
- [2025-2-5] **[ComfyUI](https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged) now supports Lumina-Image 2.0!** 🎉 Thanks to **ComfyUI**[@ComfyUI](https://github.com/comfyanonymous/ComfyUI)! 🙌 Feel free to try it out! 🚀
- [2025-1-31] We have released the latest .pth format weight file [Hugging Face](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/tree/main).
- [2025-1-25] 🚀🚀🚀 We are excited to release `Lumina-Image 2.0`, including:
  - 🎯 Checkpoints, Fine-Tuning and Inference code.
  - 🎯 Website & Demo are live now! Check out the [Huiying](https://magic-animation.intern-ai.org.cn/image/create) and [Gradio Demo](http://47.100.29.251:10010/)!



## 📑 Open-source Plan

 - [x] Inference 
 - [x] Checkpoints
 - [x] Web Demo (Gradio)
 - [x] Finetuning code
 - [x] ComfyUI
 - [x] Diffusers
 - [x] LoRA
 - [x] Technical Report
 - [ ] Unified multi-image generation
 - [ ] Control
 - [ ] PEFT (LLaMa-Adapter V2)

## 🎥 Demo



<div align="center">
  <video src="https://github.com/user-attachments/assets/b1d6dddf-4185-492d-b804-47d3d949adb5" width="70%"> </video>
</div>

## 🎨 Qualitative Performance

![Qualitative Results](assets/Demo.png)





## 📊 Quantitative Performance
![Quantitative Results](assets/quantitative.png)


## 🎮 Model Zoo


| Resolution | Parameter| Text Encoder | VAE | Download URL  |
| ---------- | ----------------------- | ------------ | -----------|-------------- |
| 1024       | 2.6B             |    [Gemma-2-2B](https://huggingface.co/google/gemma-2-2b)  |   [FLUX-VAE-16CH](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae) | [hugging face](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) |

## 💻 Finetuning Code
### 1. Create a conda environment and install PyTorch
```bash
conda create -n Lumina2 -y
conda activate Lumina2
conda install python=3.11 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
### 2.Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Install flash-attn
```bash
pip install flash-attn --no-build-isolation
```
### 4. Prepare data
You can place the links to your data files in `./configs/data.yaml`. Your image-text pair training data format should adhere to the following:
```json
{
    "image_path": "path/to/your/image",
    "prompt": "a description of the image"
}
```
### 5. Start finetuning
```bash
bash scripts/run_1024_finetune.sh
```
## 🚀 Inference Code
We support multiple solvers including Midpoint Solver, Euler Solver, and **DPM Solver** for inference.
> [!Note]
> Both the Gradio demo and the direct inference method use the .pth format weight file, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LQLh9CJwN3GOkS3unrqI0K_q9nbmqwBh?usp=drive_link).

> [!Note]
> You can also directly download from [huggingface](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/tree/main). We have uploaded the .pth weight files, and you can simply specify the `--ckpt` argument as the download directory.
- Gradio Demo
```python   
python demo.py \
    --ckpt /path/to/your/ckpt \
    --res 1024 \
    --port 12123
``` 


- Direct Batch Inference
```bash
bash scripts/sample.sh
```

## Citation

If you find the provided code or models useful for your research, consider citing them as:

```bib
@misc{lumina2,
    author={Qi Qin and Le Zhuo and Yi Xin and Ruoyi Du and Zhen Li and Bin Fu and Yiting Lu and Xinyue Li and Dongyang Liu and Xiangyang Zhu and Will Beddow and Erwann Millon and Victor Perez,Wenhai Wang and Yu Qiao and Bo Zhang and Xiaohong Liu and Hongsheng Li and Chang Xu and Peng Gao},
    title={Lumina-Image 2.0: A Unified and Efficient Image Generative Framework},
    year={2025},
    eprint={2503.21758},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/pdf/2503.21758}, 
}
```


