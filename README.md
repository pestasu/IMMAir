## The main idea of iMMAir.
Predicting air quality using joint multimodal data is crucial to comprehensively capture the diverse factors influencing atmospheric conditions. Therefore, this study introduces a new multimodal learning framework that integrates outdoor images with traditional site data to improve the accuracy and reliability of air quality predictions. However, aligning and fusing these heterogeneous data remains a formidable challenge, further exacerbated by pervasive data incompleteness issues. In this paper, we propose a novel incomplete multimodal learning approach (iMMAir) to impute missing data for robust air quality prediction. Specifically, we first design a shallow feature extractor to capture modal-specific features within the embedded space.
Then we develop a conditional diffusion-driven modality recovery approach to mitigate the distribution gap between the recovered and true data. This approach infers the distribution of absent data based upon a conditional probability framework, while further integrating constraints from temporal correlations and semantic consistencies for modality completion. Finally, we reconstruct incomplete modalities and integrate available data to predict downstream tasks via a multimodal transformer network. To alleviate the modality imbalance problem, we develop an adaptive gradient modulation strategy to adjust the optimization of each modality. In experiments, iMMAir reduces prediction errors by an average of 5.6% and 2.5% compared to baseline models in terms of air quality regression and classification, respectively.
## Usage

### Prerequisites
- Python 3.8
- PyTorch 2.0.1
- CUDA 11.7

