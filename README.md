<p align="center">
  <img src="SAPSAL_logo_1.png" alt="SAPSAL Logo" width="250"/>
</p>

# SAPSAL

**SAPSAL (Star And Protoplanetary disk Spectroscopic data AnaLyzer with Neural Networks)** is a deep learning toolkit for analysing optical stellar spectra observed with VLT/MUSE using conditional invertible neural networks (cINNs). SAPSAL is named after a Korean dog breed, the SAPSAL dog.

SAPSAL networks give a full posterior distribution of stellar parameters (e.g. $$T_{\rm{eff}}$$, $$\rm{log} g$$,  $$A_{\rm{V}}$$,  $$r_{\rm{veil}}$$) from MUSE spectrum (4750 - 9350Å)

This repository provides Python codes to build and run the SAPSAL networks introduced in the following papers:
- [Kang et al. 2023](https://www.aanda.org/articles/aa/full_html/2023/06/aa46345-23/aa46345-23.html)
- [Kang et al. 2025](https://www.aanda.org/articles/aa/full_html/2025/05/aa50394-24/aa50394-24.html)
- Kang et al. 2025b, _in prep._


## Installation requirements
The following packages are required to run the scripts in this repository:

| Package | Version | Comment |
|---------|---------|---------|
| python  | >= 3.11.7   |  
| numpy   |  >= 1.26  | 
| Pytorch | >= 2.2.2  | need torch and torchvision, get [here](https://pytorch.org/) |
| pandas   | >= 2.1.4 |
| tqdm | >= 4.65.0 |
| matplotlib | >= 3.8.0 |
| GPUtil | >= 1.4.0 | not necessary, only for automatic CUDA GPU search and selection, <br> get [here](https://github.com/anderskm/gputil) |
| KDEpy | >= 1.1.9 | for MAP calcuation, get [here](https://kdepy.readthedocs.io/en/latest/index.html) |
| scipy | >= 1.11.4 |
| astropy | >= 5.3.4 |
| scikit-learn | >= 1.2.2 |
> ⚠️ Note: The versions listed are the ones tested with this repository.  
> Earlier versions may also work, but compatibility is not guaranteed.

**FrEIA** package is already included in this repository. The FrEIA used in this repository is based on FrEIA v0.2, but is not perfectly the same as the one in [FrEIA](https://github.com/vislearn/FrEIA).





## How to use pre-trained networks

You can download pre-trained networks in networks/. Currently (2025. 06), there are two networks available.

| Network name | Parameters to predict | Comments |
|---------|---------|---------|
| Stl_TGA_tpl | Teff, log g, Av (no veiling)  |  cINN trained only on BT-Settl, presented in [Kang et al. 2023](https://www.aanda.org/articles/aa/full_html/2023/06/aa46345-23/aa46345-23.html) <br>Trained on fixed Rv value of 4.4 |
| SpD_TGARL_Noise_mMUSE  |  Teff, log g, Av, r_veil, library flag  | cINN presented in [Kang et al. 2025](https://www.aanda.org/articles/aa/full_html/2025/05/aa50394-24/aa50394-24.html) <br>Trained on BT-Settl and Dusty<br>Trained on fixed Rv value of 4.4|

**SpD_TGARL_Noise_mMUSE** is the recent version used to analyse stars in Trumpler 14. It considers the flux errors in prediction process (i.e. Noise-Net mode). THis network requires MUSE sectrum and medium flux error ($$\sigma_{\rm{med}}$$ = N/S ratio) alomg the wavelength. More details about the networks are in [Kang et al. 2025](https://www.aanda.org/articles/aa/full_html/2025/05/aa50394-24/aa50394-24.html).

In each network directory, you will find a configuration file (c_XXXX.py) and a zipped network (XXXX.pt.zip). First, unzip the network to get the network file (XXXX.pt). Please keep the configuration file and network file in the same directory (this is not necessary, but it makes reading the network a bit easier).

You can find the jupyter notebook (Tutorial.ipynb) in examples/, which explains
- how to read the network
- how to prepare input data from your observations
- how to run the network and make predictions
- how to use useful functions in expander.py to calculate MAP estimates, plot posterior distributions, etc.

### MUSE wavelength
Networks are designed for the VLT/MUSE spectrum. We masked some spectral bins in the network to avoid emission lines, but used most of the spectral bins. You can find out how to filter out unnecessary spectral bins in Tutorial.ipynb.

MUSE wavelength for the whole range:
> np.arange(4750.1572265625, 9351.4072265625, 1.25)


  


<!--
## 필요한 내용
- 몇가지 훈련된 네트워크 세트 (pt파일, config)
- 훈련된 네트워크 사용방법 알려주는 주피터 노트북: 패키지 경로설정, 클래스 활용 기본, 포스테리어 얻기, MAP-unc 등 계산, 그림그리는 툴
- 임시로 MUSE 스펙트럼 파일 하나 예시. (example directory가 필요)
-->

## Citation
If you use SAPSAL networks in your work, please cite the papers below.
- [Kang et al. 2023](https://www.aanda.org/articles/aa/full_html/2023/06/aa46345-23/aa46345-23.html)
- [Kang et al. 2025](https://www.aanda.org/articles/aa/full_html/2025/05/aa50394-24/aa50394-24.html)
- Kang et al. 2025b, _in prep._
- for all networks: [Ardizzone et al. 2019b](https://arxiv.org/abs/1907.02392), [Ardizzone et al. 2021](https://arxiv.org/abs/2105.02104)


## License

This project is licensed under the MIT License.

It includes code from the FrEIA library (https://github.com/vislearn/FrEIA),  
which is also licensed under the MIT License.


<!--
## Features

- Conditional Invertible Neural Network (cINN) implementation based on [FrEIA](https://github.com/vislearn/FrEIA)
- Domain adaptation through adversarial training
- Modular model definitions (`models/`)
- Tools for data pre-processing, training, and evaluation

## Folder Structure



-->

