## The main idea of iMMAir.

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
Before running missing cases, you should [download](https://drive.google.com/drive/folders/11jeaq5x8ywgAGT0uW5yOQz6ePYamlrG7?usp=drive_link) the weights pretrained by complete multimodal data of air quality(i.e., MR=0.0).
You can put the downloaded weights into `pt/` directory.

### Run the Codes
Running the following command:
```
bash ./exp/exp_mmair.sh <config.json>
```
