# INFO6105-Final-Project: Advanced Data Science Implementations

This repository contains implementations of two cutting-edge research papers as part of the INFO6105 Data Science Engineering Methods and Tools course. Our team has successfully implemented and enhanced these approaches, focusing on graph-based knowledge retrieval and time series classification using vision transformers.

## Table of Contents
- [Project 1: LightRAG Implementation](#project-1-lightrag-implementation)
- [Project 2: ViTST Implementation](#project-2-vitst-implementation) 
- [Documentation](#documentation)
- [Team Members](#team-members)

---

## Project 1: LightRAG Implementation

### LightRAG: Improved Implementation

This repository contains an enhanced implementation of LightRAG (Light Retrieval-Augmented Generation), a simple and fast approach to retrieval-augmented generation as described in the original paper.

### Overview

LightRAG provides an efficient framework for enhancing large language models with external knowledge. This implementation extends the original approach with significant improvements to the document processing pipeline, specifically through an enhanced content-aware chunking strategy.

### Key Features

- **Adaptive Content-Aware Chunking**: Enhanced document processing that preserves semantic integrity
- **Structure-Preserving Processing**: Respect for paragraph and sentence boundaries during chunking
- **Improved Retrieval Quality**: Better context preservation for more accurate information retrieval
- **Cross-Parameter Consistency**: Stable performance across different chunk size configurations

### Installation

```bash
# Clone the repository
git clone https://github.com/Pagar-Bhagyashri/INFO6105-Final-Project.git
cd INFO6105-Final-Project/LightRAG-Implementation

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Basic Usage

```python
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import asyncio

# Define embedding function
async def embedding_func(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

# Define LLM function
async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    # Your LLM implementation here
    pass

# Initialize LightRAG
async def initialize():
    rag = LightRAG(
        working_dir="./my_lightrag",
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    return rag

# Use LightRAG
rag = asyncio.run(initialize())

# Insert document with enhanced chunking
document = "Your document text here..."
rag.insert(document)

# Query the system
response = rag.query(
    query="Your question here?",
    param=QueryParam(mode="hybrid", top_k=3),
)
print(response)
```

#### Enhanced Chunking

Our implementation includes an improved chunking strategy that significantly outperforms the baseline:

```python
# Using the enhanced chunking directly
from lightrag.improved import adaptive_chunking

chunks = adaptive_chunking(text, chunk_size=1000)
```

### Performance Improvements

Our enhanced implementation achieves:

- **96.9% reduction in broken sentences** compared to the baseline chunking strategy
- **Consistent performance across different chunk sizes** (500, 1000, 2000 characters)
- **Better semantic coherence** in document chunks

### Evaluation Results

Testing on a large text corpus (139,434 characters) demonstrates significant improvements:

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| Broken Sentences (500 char) | 277 | 5 | 98.2% |
| Broken Sentences (1000 char) | 138 | 5 | 96.4% |
| Broken Sentences (2000 char) | 69 | 5 | 92.8% |
| Average | 161.3 | 5 | 96.9% |

### Visualization and Analysis

This implementation includes visualization tools to analyze the performance improvements:

![Chunking Comparison](LightRAG-Implementation/chunking_evaluation_results.png)

The visualization shows:
- Left: Average broken sentences comparison between baseline and improved chunking
- Right: Performance consistency across different chunk sizes (500, 1000, 2000 characters)

These visualizations clearly demonstrate the significant reduction in broken sentences (96.9% improvement) and the consistent performance of our enhanced chunking strategy regardless of chunk size configuration.

### Comparison to Original Paper

Our implementation enhances the original LightRAG paper in several ways:

1. **Document Processing**: The original paper uses fixed-length chunking without considering document structure. Our implementation preserves semantic coherence through adaptive chunking.

2. **Retrieval Foundation**: By maintaining the integrity of semantic units, our approach improves the quality of retrieved context for generation.

3. **Performance Metrics**: We introduce quantitative metrics to measure chunking quality, demonstrating a 96.9% reduction in content fragmentation.

4. **Architecture Compatibility**: Our enhancements are fully compatible with the original LightRAG architecture, requiring no changes to other system components.

### Paper Information
- **Title**: "LightRAG: Simple and Fast Retrieval-Augmented Generation"
- **Authors**: Guo, Z., Xia, L., Yu, Y., Ao, T., & Huang, C. (2024)
- **arXiv**: [arXiv:2410.05779](https://arxiv.org/abs/2410.05779)

---

## Project 2: ViTST Implementation

# ViTST - Vision Transformer for Irregularly Sampled Time Series

This project implements the ViTST (Vision Transformer for Irregularly Sampled Time Series) pipeline for classifying activity types in multivariate time series data using deep learning. The model converts each time series into an image and leverages a Swin Transformer for classification.

### Overview

Traditional deep learning models struggle with **irregularly sampled time series**. This project addresses that limitation by:
- Converting time series into **line plot images** using grid-based visualization
- Using a **pre-trained Vision Transformer (Swin Transformer)** to classify the images
- Applying the approach to the **PAMAP2 dataset**, a wearable sensor dataset for human activity recognition

This approach shows superior performance compared to specialized methods, especially in handling missing observations, achieving significant improvements in classification accuracy and F1 score.

### Paper Information

This implementation is based on the paper:
- **Title**: "Time Series as Images: Vision Transformer for Irregularly Sampled Time Series"
- **Authors**: Zekun Li, Shiyang Li, Xifeng Yan
- **Published in**: NeurIPS 2023 (Conference on Neural Information Processing Systems)
- **Paper URL**: [arXiv:2303.12799](https://arxiv.org/abs/2303.12799)

### Dataset

**PAMAP2 Physical Activity Monitoring Dataset**
- Contains data from 9 subjects performing 18 different physical activities
- Each data point includes sensor readings over time for 17 variables
- Features IMU sensor data from 3 body locations (wrist, chest, ankle)
- Approximately 60% missing data, making it ideal for testing irregular time series methods
- Data is pre-processed and stored as pickle files in the `processed_data` folder

### Model Architecture

1. **TimeSeriesDataset**: Loads data and labels from the `processed_data` folder.
2. **ViTSTDataset**: Converts each multivariate time series sample into a 2D image:
   - Uses a grid layout (4×5 for PAM dataset with 17 variables)
   - Each grid cell is a line graph sized at 64×64 pixels
   - Includes markers to indicate observed data points
   - Uses linear interpolation between points
   - Employs distinct colors for different variables
   - Variables are sorted by missing ratio for better organization

3. **ViTST Model**: 
   - Uses a Swin Transformer pre-trained on ImageNet-21K
   - Features patch size 4 and window size 7
   - Fine-tuned with learning rate 2e-5
   - Trained for 20 epochs for the PAM dataset
   - Uses batch size of 72 for training

4. **ViTSTTrainer**: Handles training, evaluation, visualization, and model checkpointing.

### Implementation Details

Key implementation choices:
- **Grid Layout**: 4×5 grid for the PAM dataset with 17 variables
- **Visualization Elements**:
  - Markers (*) to indicate observed data points
  - Linear interpolation between points
  - Distinct colors for different variables
  - Variables sorted by missing ratio for better visualization

- **Model Selection**:
  - Default: Swin Transformer with patch size 4 and window size 7
  - Alternative options: ViT and ResNet for comparison
  - Models pre-trained on ImageNet-21K for transfer learning benefits

### Usage Instructions

#### Step 1: Setup
```bash
cd INFO6105-Final-Project/ViTST-Implementation
pip install -r requirements.txt
```

#### Step 2: Prepare your data
Place your processed PAM dataset files in the `processed_data` directory or modify the config to point to your data location.

#### Step 3: Run the notebook or script
```bash
# Recommended
jupyter notebook Time.ipynb

# Or if converted to script
python Time.py
```

#### Configuration
```python
config = {
  'data_path': './processed_data',
  'image_size': (256, 320),
  'grid_layout': (4, 5),
  'batch_size': 32,
  'num_workers': 2,
  'learning_rate': 2e-5,
  'num_epochs': 10,
  'device': 'cuda' if torch.cuda.is_available() else 'cpu',
  'save_dir': './'
}
```

### Results

The ViTST approach demonstrates exceptional performance on the PAM dataset:
- **Accuracy**: 95.8% (compared to 88.5% for the best baseline)
- **F1 score**: 96.5% (compared to 89.8% for the best baseline)

Especially impressive is the robustness to missing observations:
| Missing Ratio | ViTST F1 | Baseline F1 | Improvement |
|---------------|----------|-------------|-------------|
| 10%           | 93.7%    | 75.2%       | +18.5%      |
| 30%           | 87.6%    | 48.4%       | +39.2%      |
| 50%           | 80.8%    | 38.0%       | +42.8%      |

---

## Documentation

For detailed explanations of our implementations and evaluation results, please refer to our comprehensive project report:

- [Group 4 INFO6105 Final Project Report (PDF)](https://github.com/Pagar-Bhagyashri/INFO6105-Final-Project/blob/main/Group_4_INFO6105_Final_Project_Report.pdf)

This report includes:
- Detailed analysis of both research papers
- Step-by-step explanation of our implementation approaches
- Comprehensive evaluation methodologies and results
- Comparisons with baseline performance metrics
- Discussion of implications and future directions

### Future Work

Potential areas for future enhancement include:

#### LightRAG Enhancements:
1. **Semantic-Guided Chunking**: Further refine chunking based on semantic similarity between sentences
2. **Multi-Document Knowledge Linking**: Improve connections between chunks across different documents
3. **Dynamic Retrieval Optimization**: Automatically adjust retrieval parameters based on query characteristics

#### ViTST Enhancements:
1. **Self-supervised Pre-training**: Using masked image modeling for time series
2. **Alternative Visualization Methods**: Exploring beyond basic line graphs
3. **Hierarchical Attention**: Better capturing temporal dynamics in time series data

## Team Members
- Bhagyashri Avinash Pagar
- Chethan M Chandrashekar
- Divya Teja Mannava

## Repository Structure
```
INFO6105-Final-Project/
├── README.md                        # Main project README
├── LightRAG-Implementation/         # LightRAG project
│   ├── lightrag/                    # Core implementation
│   ├── examples/                    # Example usage
│   ├── evaluation/                  # Evaluation results
│   └── README.md                    # LightRAG specific README
├── ViTST-Implementation/            # ViTST project
│   ├── Time.ipynb                   # Main implementation notebook
│   ├── processed_data/              # Dataset files
│   ├── models/                      # Saved model checkpoints
│   ├── results/                     # Results and visualizations
│   └── README.md                    # ViTST specific README
└── Group_4_INFO6105_Final_Project_Report.pdf  # Complete project report
```

## References

### LightRAG References
1. Guo, Z., Xia, L., Yu, Y., Ao, T., & Huang, C. (2024). LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv preprint. https://arxiv.org/abs/2410.05779
2. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. arXiv preprint. https://arxiv.org/abs/2005.11401
3. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2023). Retrieval-augmented generation for large language models: A survey. arXiv preprint. https://arxiv.org/abs/2312.10997

### ViTST References
1. Li, Z., Li, S., & Yan, X. (2023). Time Series as Images: Vision Transformer for Irregularly Sampled Time Series. Advances in Neural Information Processing Systems, 36.
2. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10012-10022.
3. Reiss, A., & Stricker, D. (2012). Introducing a new benchmarked dataset for activity monitoring. In 2012 16th International Symposium on Wearable Computers (pp. 108-109). IEEE.
