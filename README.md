# SAPSAL

**SAPSAL (Star And Protoplanetary disk Spectroscopic data AnaLyzer with Neural Networks)** is a deep learning toolkit for analysing optical stellar spectra observed with VLT/MUSE using conditional invertible neural networks (cINNs). 

From optical stellar spectra, SAPSAL networks give a full posterior distribution of stellar parameters (e.g. $$T_{\rm{eff}}$$, $$\rm{log} g$$,  $$A_{\rm{V}}$$,  $$r_{\rm{veil}}$$)

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
| GPUtil | >= 1.4.0 | not necessary, only for automatic GPU search and selection, <br> get [here](https://github.com/anderskm/gputil) |
| KDEpy | >= 1.1.9 | for MAP calcuation, get [here](https://kdepy.readthedocs.io/en/latest/index.html) |
| scipy | >= 1.11.4 |
| astropy | >= 5.3.4 |
| scikit-learn | >= 1.2.2 |
> ⚠️ Note: The versions listed are the ones tested with this repository.  
> Earlier versions may also work, but compatibility is not guaranteed.

**FrEIA** package is already included in this repository. The FrEIA used in this repository is based on FrEIA v0.2, but is not perfectly the same as the one in [FrEIA](https://github.com/vislearn/FrEIA).





## How to use pre-trained networks
To write
- a list of neural networks available + relevant citation
- tutorial codes: Prediction (load network, predict, MAP, plots)


<!--
## 필요한 내용
- 몇가지 훈련된 네트워크 세트 (pt파일, config)
- 훈련된 네트워크 사용방법 알려주는 주피터 노트북: 패키지 경로설정, 클래스 활용 기본, 포스테리어 얻기, MAP-unc 등 계산, 그림그리는 툴
- 임시로 MUSE 스펙트럼 파일 하나 예시. (example directory가 필요)
-->

## Citation
If you use SAPSAL in your work, please cite the papers below.
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

