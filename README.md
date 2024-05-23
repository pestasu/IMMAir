## The main idea of iMMAir.
Predicting air quality using joint multimodal data is crucial to comprehensively capture the diverse factors influencing atmospheric conditions. Therefore, this study introduces a new multimodal learning framework that integrates outdoor images with traditional site data to improve the accuracy and reliability of air quality predictions. However, aligning and fusing these heterogeneous data remains a formidable challenge, further exacerbated by pervasive data incompleteness issues. In this paper, we propose a novel incomplete multimodal learning approach (iMMAir) to impute missing data for robust air quality prediction. Specifically, we first design a shallow feature extractor to capture modal-specific features within the embedded space.
Then we develop a conditional diffusion-driven modality recovery approach to mitigate the distribution gap between the recovered and true data. This approach infers the distribution of absent data based upon a conditional probability framework, while further integrating constraints from temporal correlations and semantic consistencies for modality completion. Finally, we reconstruct incomplete modalities and integrate available data to predict downstream tasks via a multimodal transformer network. To alleviate the modality imbalance problem, we develop an adaptive gradient modulation strategy to adjust the optimization of each modality. In experiments, iMMAir reduces prediction errors by an average of 5.6% and 2.5% compared to baseline models in terms of air quality regression and classification, respectively.
## Usage

### Prerequisites
- Python 3.8
- PyTorch 2.0.1
- CUDA 11.7

### Datasets
(1) Data is from [here](https://dl.acm.org/doi/abs/10.1145/3191752), and you can [download](https://drive.google.com/drive/folders/1onxwxoWCcORa51cxp5VIbFU7GzT0dO-m?usp=drive_link) preprocessed data files.
(2) Data is crawl down from [here](https://www.tour-beijing.com/real_time_weather_photo/).
You can put the downloaded datasets into `dataset/` directory.

### Pretrained weights
Before running missing cases, you can [download](https://drive.google.com/drive/folders/11jeaq5x8ywgAGT0uW5yOQz6ePYamlrG7?usp=drive_link) the weights pretrained by complete multimodal data of air quality(i.e., MR=0.0).
You can put the downloaded weights into `pt/` directory.

### Run the Codes
Running the following command:
```
bash ./exp/exp_mmair.sh <config.json>
```
