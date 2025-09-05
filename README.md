# **Phy-FusionNet: Official PyTorch Implementation**

This repository contains the official PyTorch implementation of the paper **"Phy-FusionNet: A Memory-Augmented Transformer for Multimodal Emotion Recognition with Periodicity and Contextual Attention"**.  

**Note:** This paper has been accepted by IEEE Transactions on Affective Computing (TAFFC) and is forthcoming.  

Phy-FusionNet is an innovative memory-augmented Transformer architecture designed for multimodal emotion recognition from physiological signals. It achieves state-of-the-art performance on multiple public datasets by effectively capturing long-term dependencies and inherent periodic patterns.

## **Core Features**

* **Memory Stream Module**: Introduces a memory module with a FIFO queue and a decay-based update mechanism to effectively preserve and utilize long-term contextual information, which is crucial for tracking gradually evolving emotional states.  
* **Fourier Analysis Module**: Integrates Fourier-based positional encoding and frequency-aware attention, enabling the model to robustly detect periodic emotional cues inherent in physiological signals.  
* **Adaptive Temporal Attention**: Employs a Mixture-of-Head (MOH) attention mechanism, allowing the model to dynamically focus on the most relevant features, thereby improving computational efficiency and the effectiveness of feature extraction.  
* **Multimodal Binding & Fusion**: Uses a Transformer-based fusion framework that balances and integrates modality-specific and shared features to generate a more comprehensive emotional representation.  
* **Automated Hyperparameter Optimization**: Built-in integration with the [Optuna](https://optuna.org/) framework for automatic Bayesian hyperparameter search to find the optimal model configuration and training strategy.

## **Environment Setup**

Please follow the steps below to set up your local environment to run the code.  
**1\. Clone the Repository**  
git clone [https://github.com/AffectiveComputingXJTLU/Phy\-FusionNet](https://github.com/AffectiveComputingXJTLU/Phy-FusionNet)

cd Phy-FusionNet

**2\. Create and Activate a Virtual Environment (Recommended)**  
python \-m venv venv  
source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3\. Install Dependencies  
The required dependencies for the project are listed in the requirements.txt file.  
pip install \-r requirements.txt

Key dependencies include:

* torch  
* pandas  
* numpy  
* scikit-learn  
* imbalanced-learn  
* optuna  
* matplotlib  
* seaborn  
* tensorboard

## **Usage Guide**

### **1\. Data Preparation**

Please place your preprocessed dataset file (e.g., merged\_data.csv) in the root directory of the project, or update the PHYSIO\_FILE variable path in the script.  

### **2\. Run Training and Evaluation**

Simply run the main script to start the entire workflow, which includes data processing, hyperparameter optimization, final model training, and evaluation.  
python phy\_fusionnet\_train.py

The script will perform the following actions:

* **Data Loading and Preprocessing**: Loads the CSV, applies standardization, and uses SMOTE to handle class imbalance in the training set.  
* **Hyperparameter Optimization**: Runs a series of trials using Optuna to find the best model configuration (default: 25 trials, max 2 hours).  
* **Final Model Training**: Trains the model on the full training set using the best hyperparameters found, and saves the best-performing model using an early stopping mechanism.  
* **Evaluation**: Evaluates the final model's performance on the test set and generates a detailed classification report and confusion matrix.

### **3\. Viewing Results**

All output files will be saved in the experiment\_output/ directory, including:

* experiment\_console.log: Complete console output log.  
* experiment\_summary.json: A JSON summary file containing all configurations, results, and artifact paths.  
* final\_best\_model.pth: The trained weights of the best-performing model.  
* optuna\_history.png / optuna\_importance.png: Visualization charts from the hyperparameter optimization process.  
* confusion\_matrix.png: The confusion matrix plot from the test set evaluation.  
* runs/: The TensorBoard log directory, which can be used to monitor the training process in real-time.

To launch TensorBoard, run:  
tensorboard \--logdir=experiment\_output/runs

## **Performance Highlights**

Our model achieves State-of-the-Art (SOTA) performance on five major public datasets for physiological signal-based emotion recognition.

* **PPB-Emo (7-Class)**: Achieved an accuracy of **98.7%**, a significant improvement of **16.3%** over the previous state-of-the-art model.  
* **CL-Drive (3-Class)**: Achieved an accuracy of **88.6%**, demonstrating excellent robustness on noisy, real-world driving scenario data.  
* **WESAD (3-Class)**: Achieved an accuracy of **99.1%**.

The model demonstrates stable performance across various emotion classes, with F1-Score differences typically below 2.5%. This indicates robust recognition capabilities, even for subtle or confusable emotions.

## **How to Cite**

If you use this project or our paper in your research, please cite our work:  

## **License**

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
