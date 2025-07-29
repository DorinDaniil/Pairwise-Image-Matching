# Pairwise-Image-Matching
![figure_1_page-0001](https://github.com/user-attachments/assets/1d035f65-e20c-41e5-95e7-bb089d96e6b7)

This work addresses the critical problem of detecting near-duplicate images in scientific publications, particularly in medical and biological research. The core challenge is to determine whether two given images are:

1. **Near-duplicates**: One image was derived from the other through manual transformations (true plagiarism), or
2. **Merely similar**: Two distinct images sharing visual content but not derived from the same source

## Why This Matters

Traditional image retrieval systems often fail to distinguish between these two scenarios. While content-based image retrieval can identify visually similar candidates, it cannot determine if one image was manually manipulated from another - a crucial distinction for plagiarism detection in academic publishing.

## Near-Duplicate Transformations

Our system detects images that have undergone common manual manipulations including:
- Rotations and mirroring
- Grayscale conversion
- Contrast adjustments
- Cropping and resizing
- Blurring and noise addition
- Combinations of these transformations

## The Classification Challenge

The key difficulty lies in differentiating between:
- **Class 1 (Near-duplicate)**: Images where one was derived from the other through manual manipulation (e.g., a grayscale version of the original)
- **Class 0 (Similar but distinct)**: Images that share visual content but are fundamentally different (e.g., two different cells under microscope)

Unlike general image similarity tasks, our goal is not to measure visual resemblance, but to detect whether one image was specifically derived from another through manual transformations - a critical distinction for plagiarism detection in scientific contexts.

## Proposed Solution
![figure_2_page-0001](https://github.com/user-attachments/assets/67dfe62a-b288-4756-aafc-d76cd31b309f)

We implement a siamese neural network architecture with:
- EfficientNet-B3 as the feature encoder
- A difference-based fusion module to create order-invariant embeddings
- A classification head that predicts the probability of near-duplicate relationship

The system outputs a similarity score representing the probability that the second image was derived from the first through manual manipulation, rather than simply sharing visual content.
