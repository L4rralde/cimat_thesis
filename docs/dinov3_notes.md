## DINOv3 Notes.

### 3. Training at scale without supervision

#### Learning objective

Following DINOv2 (Oquab et al., 2024), we use an image-level objective (Caron et al., 2021) LDINO, and balance it with a patch-level latent reconstruction objective (Zhou et al., 2021) LiBOT. We also replace the centering from DINO with the Sinkhorn-Knopp from SwAV (Caron et al., 2020) in both objectives. Each objective is computed using the output of a dedicated head on top of the backbone network, allowing for some specialization of features before the computation of the losses. Additionally, we use a dedicated layer normalization applied to the backbone outputs of the local and global crops. Empirically, we found this change to stabilize ImageNet kNNclassification late in training (+0.2 accuracy) and improve dense performance (e.g. +1 mIoU on ADE20k segmentation, -0.02 RMSE on NYUv2 depth estimation).

#### Updated model architecture

- **Model size**: We increase the size of the model to 7B parameters, and provide in Tab. 2 a comparison of the corresponding hyperparameters with the 1.1B parameter model trained in the DINOv2 work.

- **RoPE positional Embedding**: We also employ a custom variant of RoPE: our base implementation assigns coordinates in a normalized [−1, 1] box to each patch, then applies a bias in the multi-head attention operation depending on the relative position of two patches.

- **RoPE-box jittering**: To improve the robustness of the model to resolutions, scales and aspect ratios, we employ RoPE-box jittering. The coordinate box [−1, 1] is randomly scaled to [−s, s], where s ∈ [0.5, 2].


#### Optimization

Because the interplay between model capacity and training data complexity is hard to assess a priori, it
is impossible to guess the right optimization horizon. To overcome this, we get rid of all parameter scheduling, and train with constant learning rate, weight decay, and teacher EMA momentum.


### 4. Gram anchoring: A regularization for dense features.

FUTURE

As training progresses, the performance degrades on dense tasks (Figs. 5b and 5c). This phenomenon, which is due to the emergence of patch-level inconsistencies in feature representations, undermines the interest behind extended training. In this section, we first analyze the loss of patch-level consistency, then propose a new objective to mitigate it, called Gram anchoring.

### 5. Post-Training

#### Model distillation

Our distillation approach uses the same training objective as in the first training phase, ensuring consistency in learning signals. However, instead of relying on an exponential moving average (EMA) of model weights, we use the 7B model directly as the teacher to guide the smaller student models. In this case, the teacher model is fixed. We do not observe patch-level consistency issues and therefore do not apply the Gram anchoring technique.

### 6. Results.

#### Instance Recognition

We adopted a non-parametric retrieval approach. Here, database images are ranked by their cosine similarity to a given query image, using the output CLS token. Retrieval effectiveness is quantified using mean average precision for Oxford, Paris, and AmsterTime, and global average precision for Met.

They compared results with DINOv3 against self-supervised and weakly supervised models only. DINOv3 performs the best.

#### Visual Geometry Grounded Transformer with DINOv3

VGGT with DINOv3 outperforms original VGGT (with DINOv2).

### 7 Evaluating the Full Family of DINOv3 Models

convxnets are good for constrained devices optimized for convolutions, however, they degrade when resolution scales. Also, convxnets don't use a cls token, so, the interpreting the knowledge transfer is not trivial.

SigLip 2 (google) and perception encoder (meta) outperforms dinov3.txt (meta) for image + text features.


### 10 Conclusion

The progress made with DINOv3 is a testament to the promise of self-supervised learning in advancing the state of the art in computer vision and beyond.


## Appendix

### A Artifacts and Outliers in Large-Scale Training

Outliers are typically characterized as network’s activations whose values deviate significantly from the average of their distribution. During the training of DINOv3, we identified such outliers at different levels: some occurring at the patch level and others at the feature dimension level.

#### High-Norm Patch Outliers

Patch outliers negatively affect performance in DINOv2. These outliers are primarily characterized as high-norm tokens, often located in low-information background regions of an image. These tokens are observed to play a key role in the internal communication between patches and the CLS token.

**Token registers**

To mitigate the appearance of hig-norm token outliers, we introduce additional (4) tokens, called registers, into the input sequence of the ViT. Their role is to take over the internal communication between patches and the CLS.

**Integrating Biases in the Attention Mechanism**

Two promising solutions which seem relevant and require minimal changes to the attention, specifically the explicit fixed bias, which we call ‘value gating’, and the attention bias strategies. 

Notably, the best performance is achieved with the incorporation of the register tokens, which is why we adopt this strategy for all experiments reported in the paper.


#### Feature Dimension Outliers

During the training of 7B models, we observe a distinct type of outlier that emerges not across patches, but within the feature (channel) dimension of the learned representations. A small subset of feature dimensions attain exceptionally large magnitudes, even as the norms across patches remain stable. Interestingly, these feature dimension outliers exhibit consistently high values across different patches and images. Moreover, these outlier dimensions consistently persist across the layers of a given model, increasing in magnitude with depth and reaching their maximum values in the output layer. They also progressively increase in magnitude throughout the course of training.

We conduct experiments attempting to neutralize these dimensions during both training and inference. Removing these dimensions at inference time does not lead to significant performance changes, suggesting that they primarily carry trivial or non-informative signals. Additionally, we observe that the final layer normalization is trained to substantially scale down these outlier dimensions. Thus, we recommend to apply the final layer norm to the features of the final layer for downstream use. Alternatively, applying batch normalization can also suppress these feature dimension outliers, as their elevated values are consistent across patches and images.

While the final layer normalization is well-suited to normalize the distribution of the final features, its learned parameters may be suboptimal for applying it to the features of earlier layers. Indeed, we observe performance decreases for some tasks from doing so. In these cases, we found standard feature scaling techniques (e.g. normalization with batch norm or principal component analysis) to be effective in dealing with the feature dimension outliers.

### B Additional results

#### B.2 Per layer analysis

The results are shown in Fig. 21. We find that for classification and dense tasks, performance increases smoothly over the layers. Depth estimation, tracking, and 3D correspondence estimation peak around layer 32, indicating that, for tasks where geometry plays a significant role, the downstream performance of DINOv3 can be improved by considering earlier layers. On the other hand, the performance of intermediate layers only slightly improves compared to the last one, making it a good default choice.

### C Implementation details

We train for 1M iterations using a fully-sharded data-parallel setup in Pytorch, using bfloat16 and 8-bit floating-point matrix multiplications???


### D Experimental details

#### D.8 Instance recognition

- For Oxford and Paris, we resize all images such that the larger (shorter?) side is 224 pixels long, keeping the aspect ratio, then take a full center crop, yielding an image of resolution of 224 × 224.
- For AmsterTime, we resize all images such that the shorter side is 256 pixels long, keeping the aspect ratio, then take a center crop of size 224 × 224.
- For Met, we evaluate (actually, all are evaluations) all images close to their original resolution, resizing both to the nearest multiple of patch size (resulting in a long side 508/512 for patch size 14/16).

#### D.11 Monocular depth estimation

Monocular depth estimation with DINOv3 is inspared by Depth Anything v2 (DAv2 -- DINOv2 + DPT head + something else). The backbone (DINOv3) is kept frozen throughout training.

#### D.12 Visual Geometry Grounded Transformer with DINOv3

Compared to the original VGGT, we adopt the following changes: (1) we use an image size of 592 instead of 518; this is to match the number of patch tokens that DINOv2 produces, (2) adopting a smaller learning rate, specifically from 0.0002 to 0.0001, and (3) using a concatenation of the four intermediate layers of DINOv3 ViT-L rather than just the last layer as input to the downstream modules. Interestingly, we found that using four intermediate layers brings a benefit for DINOv3, whereas doing the same for DINOv2 brings no additional performance gains. We also experimented with a version closer to the original VGGT setup (image size 512, same learning rate, final layer), and already found this untuned version to improve over the original VGGT work across all tested benchmarks.
