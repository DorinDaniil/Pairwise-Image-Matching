# Pairwise-Image-Matching
This work addresses the critical problem of detecting near-duplicate images in scientific publications, particularly in medical and biological research. The core challenge is to determine whether two given images are:

![figure_1 (4)_page-0001](https://github.com/user-attachments/assets/ce5a7d89-54b0-4eb0-82b3-42000f77de57)


- Near-duplicates: One image was derived from the other through manual transformations (true plagiarism), or
- Merely similar: Two distinct images sharing visual content but not derived from the same source.

## Why This Matters
Traditional image retrieval systems often fail to distinguish between these two scenarios. While content-based image retrieval can identify visually similar candidates, it cannot determine if one image was manually manipulated from another $-$ a crucial distinction for plagiarism detection in academic publishing.
Near-Duplicate Transformations
Our system detects images that have undergone common manual manipulations, including:

- Rotations and mirroring
- Grayscale conversion
- Contrast adjustments
- Cropping and resizing
- Blurring and noise addition
- Combinations of these transformations

## The Classification Challenge
The key difficulty lies in differentiating between:

- Class 1 (Near-duplicate): Images where one was derived from the other through manual manipulation (e.g., a grayscale version of the original).
- Class 0 (Similar but distinct): Images that share visual content but are fundamentally different (e.g., two different cells under a microscope).

Unlike general image similarity tasks, our goal is not to measure visual resemblance but to detect whether one image was specifically derived from another through manual transformations $-$ a critical distinction for plagiarism detection in scientific contexts.

## Proposed Solution

![figure_2 (3)_page-0001](https://github.com/user-attachments/assets/b044d67c-0011-4805-b5ed-a0c08cd917cd)

We implement a Siamese neural network architecture with:

Various encoders including EfficientNet-B3, ViT-L/16, CLIP ViT-H/14, and a Barlow Twins encoder using a ResNet50 backbone. Some encoders are kept frozen to compare representations obtained from our training with their contrastive encoders.
A fusion module that employs a symmetric function to ensure invariance to input order, crucial for a stable scoring function.
A classification head that predicts the probability of a near-duplicate relationship, implemented as a Multi-Layer Perceptron (MLP) with a single hidden layer, followed by ReLU activation and dropout for regularization, concluding with a sigmoid function to yield a similarity score.

The system outputs a similarity score representing the probability that the second image was derived from the first through manual manipulation, rather than simply sharing visual content.

This architecture allows us to effectively distinguish between near-duplicate and merely similar images, providing a robust solution for detecting plagiarism in scientific publications.
