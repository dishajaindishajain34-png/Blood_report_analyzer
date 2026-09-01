# Multimodal Anemia Fusion Model

*HemaVision AI — Image + CBC Integration Layer*

---

## 1. Overview

The Multimodal Anemia Fusion Model combines probabilities from two independent ML streams:
1. **Image-Based Model** (MobileNetV3-Small on blood smear) : `P(image)`
2. **CBC-Based Model** (Random Forest on Hb, MCV, MCH, MCHC) : `P(cbc)`

---

## 2. Fusion Strategy Priority

1. **Learned Logistic Regression**: Trained when paired image + CBC metadata is available.
2. **Average Fallback**: Used when both image and CBC probabilities are provided:
   ```
   P(fused) = (P(image) + P(cbc)) / 2
   ```
3. **Single-Modality Passthrough**: Used if only image or only CBC is provided.

---

## 3. Integration Architecture

```
Blood Smear Image --> MobileNetV3-Small --> P(image) --\
                                                        +--> Fusion Engine --> Final Anemia Assessment
Laboratory CBC    --> Random Forest     --> P(cbc)   --/
```
