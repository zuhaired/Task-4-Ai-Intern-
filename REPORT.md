# Fairness, Bias & Explainability: Short Write-up

**Model:** CIFAR-10 classifier, frozen MobileNetV2 + dense head (81.81% test accuracy).
**Notebook:** `fairness_analysis.ipynb`. Running it produces every plot and a `fairness_results.md` with the measured numbers to cite alongside this write-up.

## 1. Scope and an honest limitation

CIFAR-10 contains no demographic or other protected attributes, so a conventional fairness audit (e.g. by gender, race or age) cannot be done on this data. Claiming otherwise would be misleading. The audit instead checks the two kinds of disparity the data *does* support:

| Check | "Group" used | Metrics |
|---|---|---|
| Class-level bias | the 10 classes | recall (TPR), false-positive rate, precision, per-class calibration (ECE), recall gap and min/max ratio |
| Condition-slice bias | brightness, contrast, saturation terciles | accuracy per slice with 95% Wilson confidence intervals; class x brightness heat map |
| Robustness bias | darker, brighter, noise, blur, grayscale | overall accuracy, worst-class accuracy, class gap under each perturbation |

If this model were ever applied to images of people, none of these checks would be a substitute for evaluation on a dataset with labelled demographic attributes (for example FairFace) and the metrics above computed per demographic group (equal opportunity, demographic parity, calibration by group).

## 2. Explainability methods

- **Global importance:** permutation importance of the R, G, B channels, and occlusion of the image centre vs. the border per class. A class whose accuracy collapses when only the *border* is hidden is likely using background context (sky, water, road) as a shortcut instead of the object.
- **Local explanations:** SHAP (Partition explainer, blur masker) and LIME (superpixels), applied to correctly classified images and to misclassified images from the most-confused class pair. Aggregated mean |SHAP| maps show where each class's evidence sits spatially. Using two methods is a sanity check: where they disagree, neither explanation should be over-trusted.

## 3. How to read the results (what to look for)

These are hypotheses to confirm against the generated numbers, not findings.

- **Recall gap:** a wide gap between best and worst class means the model serves some classes unequally. On CIFAR-10, animal classes with similar shape and texture (cat/dog, deer/horse) and vehicle pairs (automobile/truck) are the usual weak points.
- **Calibration:** a class with high confidence but low accuracy is the most harmful kind of error, since users cannot tell it is wrong.
- **Slice gaps:** compare each accuracy gap to the confidence intervals; only gaps larger than the intervals are real.
- **Border vs. centre:** a border/centre ratio well above 1 for a class signals background reliance, which tends to fail when the context changes.
- **Training vs. validation:** training accuracy (69%) is below validation accuracy (82%) because it is measured with augmentation and dropout on, so it is not a sign of leakage, but augmentation strength should be considered when interpreting robustness results.

## 4. Mitigation recommendations (prioritised)

1. **Measure before changing anything.** Fix a target, for example "worst-class recall no more than X points below the mean", and track it with the same metrics every retrain.
2. **Class reweighting or resampling** (implemented as the optional experiment in the notebook): up-weight under-served classes using *training-set* error, never test-set error. Check that overall accuracy does not fall more than the gain in worst-class recall justifies.
3. **Fine-tune the top MobileNetV2 blocks** at a low learning rate (e.g. 1e-5 to 1e-4). With a frozen base, the head cannot learn the fine features that separate similar classes; this usually helps the hardest classes most.
4. **Reduce background shortcuts:** stronger augmentation (random crop and erasing, colour jitter, cutout) and, if available, more varied backgrounds per class. Re-run the border/centre occlusion test to confirm the ratio falls.
5. **Improve robustness slices:** add brightness, contrast, blur and noise augmentation so dark, low-contrast or noisy inputs are not systematically worse.
6. **Calibrate:** apply temperature scaling on a held-out validation set so confidence reflects accuracy; abstain or route to a human when confidence is below a threshold.
7. **Upscale inputs / higher resolution data:** the 32x32 source images limit the detail available; training on larger images (or a higher input resolution) is the largest single lever for fine-grained confusions.
8. **Monitor in production:** log per-class and per-slice accuracy and the confidence distribution, and alert on drift. Document the model's intended use and known weak classes in a model card.

## 5. Limitations of this analysis

- Slice properties (brightness, etc.) are proxies, not protected groups.
- SHAP and LIME are approximations: results depend on the masker, segmentation and sample counts, and 32x32 images give coarse superpixels. Explanations show what the model *uses*, not whether that use is *correct* or causal.
- The mitigation experiment derives weights from training-set error, which is optimistic since the model saw that data; it ranks hard classes correctly but does not give a true error estimate.
- This write-up and notebook were prepared without being executed (no TensorFlow or dataset access in the authoring environment), so it contains no measured results. Run the notebook to obtain them.
