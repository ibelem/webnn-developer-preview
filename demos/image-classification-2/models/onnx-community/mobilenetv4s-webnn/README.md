---
library_name: transformers.js
pipeline_tag: image-classification
tags:
- webnn
license: apache-2.0
---

https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

**Example:** Perform image classification with `onnx-community/mobilenetv4s-webnn`

```js
import { pipeline } from '@huggingface/transformers';

// Create an image classification pipeline
const classifier = await pipeline('image-classification', 'onnx-community/mobilenetv4s-webnn');

// Classify an image
const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
const output = await classifier(url);
// [{ label: 'tiger, Panthera tigris', score: 0.903573540929381 }]
```

---

Note: Having a separate repo for ONNX weights is intended to be a temporary solution until WebML gains more traction. If you would like to make your models web-ready, we recommend converting to ONNX using [🤗 Optimum](https://huggingface.co/docs/optimum/index) and structuring your repo like this one (with ONNX weights located in a subfolder named `onnx`).