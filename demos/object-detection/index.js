// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run Image Classification with webnn in onnxruntime-web.
//

import * as transformers from "../../assets/dist_transformers/1.21.0-dev.20241122/transformers.js";
import {
    $,
    $$,
    log,
    logError,
    getQueryValue,
    getWebnnStatus,
    updateQueryStringParameter,
    // getMedian,
    // getAverage,
    // getMinimum,
    // asyncErrorHandling,
    // getMode,
    showCompatibleChromiumVersion,
} from "../../assets/js/common_utils.js";

transformers.env.backends.onnx.wasm.proxy = false;
transformers.env.backends.onnx.wasm.simd = true;
transformers.env.backends.onnx.wasm.numThreads = 1;
transformers.env.backends.onnx.wasm.wasmPaths = "../../assets/dist_transformers/1.21.0-dev.20241122/";

const useRemoteModels = location.hostname.includes("github.io");
transformers.env.allowRemoteModels = useRemoteModels;
transformers.env.allowLocalModels = !useRemoteModels;
log("[Transformer.js] env.allowRemoteModels: " + transformers.env.allowRemoteModels);
log("[Transformer.js] env.allowLocalModels: " + transformers.env.allowLocalModels);
if (transformers.env.allowLocalModels) {
    transformers.env.localModelPath = "./";
    log("[Transformer.js] env.localModelPath: " + transformers.env.localModelPath);
}

let provider = "webnn";
let deviceType = "gpu";
let dataType = "fp16";
let modelId = "webnn/yolov8n";
let modelName = "yolov8n";
let backendLabels, modelLabels;
let label_webgpu, label_webnn_gpu, label_webnn_npu, label_yolov8n, label_yolov8m, label_yolo11;
let start, stop;
let fullResult;
// let first, average, median, best, throughput;
let status, circle, info;

let videoElement, canvasElement, overlayElement, statusElement;
let fpsElement;
let confidenceSlider, confidenceValue;

let result;
// let label1, label2, label3;
// let score1, score2, score3;
let dataTypeSpan;
let modelNameSpan;
// let latency;
let latencyDiv, indicator;
let title, device, badge;

let confidenceThreshold = 0.25;
let isProcessing = false;
let lastFrameTime = 0;
let detectionLoopId = null; // Store the requestAnimationFrame ID
let stream = null; // Store the camera stream

const COLORS = [
    "#FF3838",
    "#FF9D97",
    "#FF701F",
    "#FFB21D",
    "#CFD231",
    "#48F90A",
    "#92CC17",
    "#3DDB86",
    "#1A9334",
    "#00D4BB",
    "#2C99A8",
    "#00C2FF",
    "#344593",
    "#6473FF",
    "#0018EC",
    "#8438FF",
    "#520085",
    "#CB38FF",
    "#FF95C8",
    "#FF37C7",
];

const initializeModel = async () => {
    statusElement.textContent = "Loading YOLOv8m model...";

    fullResult.setAttribute("class", "none");
    result.setAttribute("class", "none");
    latencyDiv.setAttribute("class", "latency none");
    start.disabled = true;

    if (getQueryValue("model")) {
        modelName = getQueryValue("model");
        switch (modelName) {
            case "yolov8n":
                modelId = "webnn/yolov8n";
                break;
            case "yolov8m":
                modelId = "webnn/yolov8m";
                break;
            case "yolo11":
                modelId = "webnn/yolo11";
                break;
            default:
                modelId = "webnn/yolov8n";
                break;
        }
    }

    let device = "webnn-gpu";
    if (provider.toLowerCase() === "webnn") {
        device = `${provider}-${deviceType}`;
    } else {
        device = provider;
    }

    let options = {
        dtype: dataType,
        device: device, // 'webnn-gpu' and 'webnn-npu'
        session_options: {
            freeDimensionOverrides: {},
        },
    };

    modelNameSpan.innerHTML = dataType;
    dataTypeSpan.innerHTML = modelId;

    try {
        log("[ONNX Runtime] Options: " + JSON.stringify(options));
        log(`[Transformer.js] Loading ${modelId} and running on ${device}`);

        const model = await transformers.AutoModel.from_pretrained(modelId, options);
        let processor = await transformers.AutoProcessor.from_pretrained(modelId);

        // Configure processor to match model's expected input size (640x640)
        processor.feature_extractor.size = { width: 640, height: 640 };

        // Log the class names from the model config
        console.log("Model config:", model.config);
        console.log("Class labels:", model.config.id2label);

        statusElement.textContent = "Model loaded! Starting camera...";
        return { model, processor };

        // if (err) {
        //     status.setAttribute("class", "red");
        //     info.innerHTML = err.message;
        //     logError(err.message);
        // } else {
        //     // if (getMode()) {
        //     //     log(JSON.stringify(transformers.getPerf()));
        //     //     let warmUp = transformers.getPerf().warmup;
        //     //     let averageInference = getAverage(transformers.getPerf().inference);
        //     //     let medianInference = getMedian(transformers.getPerf().inference);
        //     //     latency.innerHTML = medianInference.toFixed(2);
        //     //     first.innerHTML = warmUp.toFixed(2);
        //     //     average.innerHTML = averageInference;
        //     //     median.innerHTML = medianInference.toFixed(2);
        //     //     best.innerHTML = getMinimum(transformers.getPerf().inference);
        //     //     throughput.innerHTML = `${transformers.getPerf().throughput} FPS`;
        //     //     fullResult.setAttribute("class", "");
        //     //     latencyDiv.setAttribute("class", "latency");
        //     // }

        //     // label1.innerHTML = output[0].label;
        //     // score1.innerText = output[0].score;
        //     // label2.innerText = output[1].label;
        //     // score2.innerText = output[1].score;
        //     // label3.innerText = output[2].label;
        //     // score3.innerText = output[2].score;
        //     // result.setAttribute("class", "");
        //     // start.disabled = false;
        //     // log(JSON.stringify(output));
        //     // log(`[Transformer.js] Classifier completed`);
        // }
    } catch (error) {
        log(`[Error] ${error}`);

        status.setAttribute("class", "red");
        info.innerHTML = `
            ${error}<br>
            Your device probably doesn't have a supported GPU.`;
        start.disabled = true;
        log(`[Error] ${error}`);
        log(`[Error] Your device probably doesn't have a supported GPU`);
    }
};

function setupCamera() {
    return new Promise((resolve, reject) => {
        navigator.mediaDevices
            .getUserMedia({
                video: {
                    facingMode: "environment",
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
            })
            .then(stream => {
                videoElement.srcObject = stream;

                videoElement.onloadedmetadata = () => {
                    // Set canvas size to 720px width and proportional height
                    const aspectRatio = videoElement.videoHeight / videoElement.videoWidth;
                    canvasElement.width = 720;
                    canvasElement.height = 720 * aspectRatio;
                    overlayElement.style.width = `${canvasElement.width}px`;
                    // overlayElement.style.height = `${canvasElement.height}px`;

                    // Start video playback
                    videoElement.play();
                    resolve(stream);
                };
            })
            .catch(error => {
                statusElement.textContent = `Camera error: ${error.message}`;
                reject(error);
            });
    });
}

function processDetections(outputs, imageWidth, imageHeight, classLabels) {
    // Process YOLOv8 outputs (shape: [1, 84, 8400])
    // For each of the 8400 predictions, we have 84 values:
    // - First 4 are bounding box coordinates (x, y, width, height)
    // - Remaining 80 are class confidences for COCO dataset

    // Clear previous detections
    overlayElement.innerHTML = "";

    const scaleX = canvasElement.width / 640; // Scale factor for width
    const scaleY = canvasElement.height / 640; // Scale factor for height

    const predictions = outputs.tolist()[0]; // Get the first batch
    const numClasses = predictions.length - 4; // Subtract 4 for bbox coordinates
    const numPredictions = predictions[0].length; // Number of predictions (8400)

    let detections = [];

    // Process each prediction
    for (let i = 0; i < numPredictions; i++) {
        const x = predictions[0][i];
        const y = predictions[1][i];
        const w = predictions[2][i];
        const h = predictions[3][i];

        let maxScore = 0;
        let maxClassIndex = -1;

        for (let c = 0; c < numClasses; c++) {
            const score = predictions[c + 4][i];
            if (score > maxScore) {
                maxScore = score;
                maxClassIndex = c;
            }
        }

        if (maxScore < confidenceThreshold) continue;

        const xmin = (x - w / 2) * scaleX;
        const ymin = (y - h / 2) * scaleY;
        const width = w * scaleX;
        const height = h * scaleY;

        detections.push({
            bbox: [xmin, ymin, width, height],
            score: maxScore,
            class: maxClassIndex,
        });
    }

    // Apply Non-Maximum Suppression (NMS)
    detections = applyNMS(detections, 0.5); // 0.5 is the IoU threshold

    // Render filtered detections
    detections.forEach(detection => {
        const [x, y, width, height] = detection.bbox;
        const className = classLabels[detection.class];
        const color = COLORS[detection.class % COLORS.length];
        const score = detection.score;

        const boxElement = document.createElement("div");
        boxElement.className = "detection-box";
        boxElement.style.left = `${x}px`;
        boxElement.style.top = `${y}px`;
        boxElement.style.width = `${width}px`;
        boxElement.style.height = `${height}px`;
        boxElement.style.borderColor = color;

        const labelElement = document.createElement("div");
        labelElement.className = "detection-label";
        labelElement.style.backgroundColor = color;
        labelElement.textContent = `${className} ${(score * 100).toFixed(1)}%`;

        boxElement.appendChild(labelElement);
        overlayElement.appendChild(boxElement);
    });

    return detections.length;
}

function applyNMS(detections, iouThreshold) {
    // Sort detections by confidence score in descending order
    detections.sort((a, b) => b.score - a.score);

    const filteredDetections = [];
    const used = new Array(detections.length).fill(false);

    for (let i = 0; i < detections.length; i++) {
        if (used[i]) continue;

        const detectionA = detections[i];
        filteredDetections.push(detectionA);

        for (let j = i + 1; j < detections.length; j++) {
            if (used[j]) continue;

            const detectionB = detections[j];
            const iou = calculateIoU(detectionA.bbox, detectionB.bbox);

            if (iou > iouThreshold) {
                used[j] = true; // Suppress overlapping box
            }
        }
    }

    return filteredDetections;
}

function calculateIoU(boxA, boxB) {
    const [xA, yA, wA, hA] = boxA;
    const [xB, yB, wB, hB] = boxB;

    const x1 = Math.max(xA, xB);
    const y1 = Math.max(yA, yB);
    const x2 = Math.min(xA + wA, xB + wB);
    const y2 = Math.min(yA + wA, yB + hB);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = wA * hA;
    const areaB = wB * hB;

    const union = areaA + areaB - intersection;
    return intersection / union;
}

async function detectFrame(model, processor, ctx) {
    if (isProcessing) return;
    isProcessing = true;

    try {
        const startTime = performance.now();

        // Capture current frame from video
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        const imageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);

        // Convert to RawImage format for transformers.js
        const image = new transformers.RawImage(imageData.data, canvasElement.width, canvasElement.height, 4);

        // Process image and run model
        const inputs = await processor(image);
        const { outputs } = await model(inputs);

        // Extract class labels from model config
        const classLabels = {};
        for (const [id, label] of Object.entries(model.config.id2label)) {
            classLabels[id] = label;
        }

        // Process and display detections
        const detectionCount = processDetections(outputs, canvasElement.width, canvasElement.height, classLabels);

        // Calculate FPS
        const endTime = performance.now();
        const frameTime = endTime - startTime;
        const fps = 1000 / (endTime - lastFrameTime);
        lastFrameTime = endTime;

        // Update status
        statusElement.textContent = `Detected: ${detectionCount} objects`;
        fpsElement.textContent = `FPS: ${fps.toFixed(1)} | Processing: ${frameTime.toFixed(0)}ms`;
    } catch (error) {
        console.error("Detection error:", error);
        statusElement.textContent = `Error: ${error.message}`;
    } finally {
        isProcessing = false;
    }
}

async function startDetection() {
    try {
        // Initialize model and camera
        const { model, processor } = await initializeModel();
        stream = await setupCamera();

        // Get the canvas context with willReadFrequently set to true
        const ctx = canvasElement.getContext("2d", { willReadFrequently: true });

        // Main detection loop
        function detectionLoop() {
            detectFrame(model, processor, ctx).finally(() => {
                detectionLoopId = requestAnimationFrame(detectionLoop);
            });
        }

        // Start detection loop
        detectionLoop();
    } catch (error) {
        console.error("Application error:", error);
        statusElement.textContent = `Failed to start: ${error.message}`;
    }
}

function stopDetection() {
    if (detectionLoopId) {
        cancelAnimationFrame(detectionLoopId);
        detectionLoopId = null;
    }

    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        stream = null;
    }

    videoElement.srcObject = null;
    isProcessing = false; // Ensure no further frames are processed
    statusElement.textContent = "Detection stopped.";
    fpsElement.textContent = "FPS: 0";
}

const checkWebNN = async () => {
    let webnnStatus = await getWebnnStatus();

    if (getQueryValue("provider")?.toLowerCase() === "webgpu") {
        circle.setAttribute("class", "none");
        info.innerHTML = "";
    }

    if (getQueryValue("provider")?.toLowerCase() === "webnn") {
        circle.setAttribute("class", "");
        if (webnnStatus.webnn) {
            status.setAttribute("class", "green");
            info.innerHTML = "WebNN supported";

            if (deviceType.toLowerCase() === "npu") {
                try {
                    await navigator.ml.createContext({ deviceType: "npu" });
                } catch (error) {
                    status.setAttribute("class", "red");
                    info.innerHTML = `
            ${error}<br>
            Your device probably doesn't have an AI processor (NPU) or the NPU driver is not successfully installed.`;
                    start.disabled = true;
                    log(`[Error] ${error}`);
                    log(
                        `[Error] Your device probably doesn't have an AI processor (NPU) or the NPU driver is not successfully installed`,
                    );
                }
            } else {
                start.disabled = false;
            }
        } else {
            if (webnnStatus.error) {
                status.setAttribute("class", "red");
                info.innerHTML = `WebNN not supported: ${webnnStatus.error} <a id="webnn_na" href="../../install.html" title="WebNN Installation Guide">WebNN Installation Guide</a>`;
                logError(`[Error] ${webnnStatus.error}`);
                indicator.innerHTML = `<a href="../../install.html" title="WebNN Installation Guide">Please set up WebNN at first</a>`;
            } else {
                status.setAttribute("class", "red");
                info.innerHTML = "WebNN not supported";
                logError(`[Error] WebNN not supported`);
            }
            start.disabled = true;
        }
    }
};

const initModelSelector = () => {
    provider = getQueryValue("provider").toLowerCase();
    deviceType = getQueryValue("devicetype").toLowerCase();
    if (provider && deviceType) {
        backendLabels.forEach(label => {
            label.setAttribute("class", "btn");
        });
        if (provider === "webgpu" && deviceType === "gpu") {
            label_webgpu.setAttribute("class", "btn active");
        } else if (provider === "webnn" && deviceType === "gpu") {
            label_webnn_gpu.setAttribute("class", "btn active");
        } else if (provider === "webnn" && deviceType === "npu") {
            label_webnn_npu.setAttribute("class", "btn active");
        }
    }

    if (getQueryValue("model")) {
        modelLabels.forEach(label => {
            label.setAttribute("class", "btn");
        });
        if (getQueryValue("model").toLowerCase() === "yolov8n") {
            label_yolov8n.setAttribute("class", "btn active");
        } else if (getQueryValue("model").toLowerCase() === "yolov8m") {
            label_yolov8m.setAttribute("class", "btn active");
        } else if (getQueryValue("model").toLowerCase() === "yolo11") {
            label_yolo11.setAttribute("class", "btn active");
        }
    }
};

const controls = async () => {
    let backendBtns = $("#backendBtns");
    let modelBtns = $("#modelBtns");

    confidenceSlider.value = confidenceThreshold;
    confidenceValue.textContent = confidenceThreshold;
    confidenceSlider.addEventListener("input", () => {
        confidenceThreshold = parseFloat(confidenceSlider.value);
        confidenceValue.textContent = confidenceThreshold.toFixed(2);
    });

    const updateBackend = e => {
        backendLabels.forEach(label => {
            label.setAttribute("class", "btn");
        });
        e.target.parentNode.setAttribute("class", "btn active");
        if (e.target.id.trim() === "webgpu") {
            let currentUrl = window.location.href;
            let updatedUrl = updateQueryStringParameter(currentUrl, "provider", "webgpu");
            window.history.pushState({}, "", updatedUrl);
            currentUrl = window.location.href;
            updatedUrl = updateQueryStringParameter(currentUrl, "devicetype", "gpu");
            window.history.pushState({}, "", updatedUrl);
            provider = "webgpu";
            deviceType = "gpu";
        } else if (e.target.id.trim() === "webnn_gpu") {
            let currentUrl = window.location.href;
            let updatedUrl = updateQueryStringParameter(currentUrl, "provider", "webnn");
            window.history.pushState({}, "", updatedUrl);
            currentUrl = window.location.href;
            updatedUrl = updateQueryStringParameter(currentUrl, "devicetype", "gpu");
            window.history.pushState({}, "", updatedUrl);
            provider = "webnn";
            deviceType = "gpu";
        } else if (e.target.id.trim() === "webnn_npu") {
            let currentUrl = window.location.href;
            let updatedUrl = updateQueryStringParameter(currentUrl, "provider", "webnn");
            window.history.pushState({}, "", updatedUrl);
            currentUrl = window.location.href;
            updatedUrl = updateQueryStringParameter(currentUrl, "devicetype", "npu");
            window.history.pushState({}, "", updatedUrl);
            provider = "webgpu";
            deviceType = "npu";
        }

        updateUi();
    };

    const updateModel = e => {
        modelLabels.forEach(label => {
            label.setAttribute("class", "btn");
        });
        e.target.parentNode.setAttribute("class", "btn active");
        if (e.target.id.trim() === "yolov8n") {
            let currentUrl = window.location.href;
            let updatedUrl = updateQueryStringParameter(currentUrl, "model", "yolov8n");
            window.history.pushState({}, "", updatedUrl);
            modelName = "yolov8n";
        } else if (e.target.id.trim() === "yolov8m") {
            let currentUrl = window.location.href;
            let updatedUrl = updateQueryStringParameter(currentUrl, "model", "yolov8m");
            window.history.pushState({}, "", updatedUrl);
            modelName = "yolov8m";
        } else if (e.target.id.trim() === "yolo11") {
            let currentUrl = window.location.href;
            let updatedUrl = updateQueryStringParameter(currentUrl, "model", "yolo11");
            window.history.pushState({}, "", updatedUrl);
            modelName = "yolo11";
        }
        updateUi();
    };

    backendBtns.addEventListener("change", updateBackend, false);
    modelBtns.addEventListener("change", updateModel, false);
};

const badgeUpdate = () => {
    if (getQueryValue("provider")?.toLowerCase() === "webgpu") {
        title.innerHTML = "WebGPU";
        provider = "webgpu";
        deviceType = "gpu";
        device.innerHTML = "GPU";
        badge.setAttribute("class", "gpu");
    } else if (getQueryValue("provider")?.toLowerCase() === "wasm") {
        title.innerHTML = "Wasm";
        provider = "wasm";
        deviceType = "cpu";
        device.innerHTML = "CPU";
        badge.setAttribute("class", "cpu");
    } else {
        title.innerHTML = "WebNN";
        provider = "webnn";
        deviceType = "gpu";
        device.innerHTML = "GPU";
        badge.setAttribute("class", "gpu");
        if (getQueryValue("devicetype")?.toLowerCase() === "cpu") {
            deviceType = "cpu";
            device.innerHTML = "CPU";
            badge.setAttribute("class", "cpu");
        } else if (getQueryValue("devicetype")?.toLowerCase() === "npu") {
            deviceType = "npu";
            device.innerHTML = "NPU";
            badge.setAttribute("class", "npu");
        } else {
            deviceType = "gpu";
            device.innerHTML = "GPU";
            badge.setAttribute("class", "gpu");
        }
    }
};

const updateUi = async () => {
    if (getQueryValue("model")) {
        modelName = getQueryValue("model");
    }

    initModelSelector();
    badgeUpdate();
    log(`[Config] Demo config updated · ${modelName} · ${provider} · ${deviceType}`);
    await checkWebNN();
    console.log(provider);
    console.log(deviceType);
    console.log(modelName);
};

const ui = async () => {
    if (!(getQueryValue("provider") && getQueryValue("model") && getQueryValue("devicetype"))) {
        let url = "?provider=webnn&devicetype=gpu&model=yolov8n";
        location.replace(url);
    }

    status = $("#webnnstatus");
    circle = $("#circle");
    info = $("#info");
    backendLabels = $$(".backends label");
    modelLabels = $$(".models label");
    label_webgpu = $("#label_webgpu");
    label_webnn_gpu = $("#label_webnn_gpu");
    label_webnn_npu = $("#label_webnn_npu");
    label_yolov8n = $("#label_yolov8n");
    label_yolov8m = $("#label_yolov8m");
    label_yolo11 = $("#label_yolo11");
    start = $("#start");
    stop = $("#stop");
    fullResult = $("#full-result");
    // first = $("#first");
    // average = $("#average");
    // median = $("#median");
    // best = $("#best");
    // throughput = $("#throughput");
    // label1 = $("#label1");
    // label2 = $("#label2");
    // label3 = $("#label3");
    // score1 = $("#score1");
    // score2 = $("#score2");
    // score3 = $("#score3");
    result = $("#result");
    title = $("#title");
    device = $("#device");
    badge = $("#badge");
    indicator = $("#indicator");
    modelNameSpan = $("#data-type");
    dataTypeSpan = $("#model-id");
    // latency = $("#latency");
    latencyDiv = $(".latency");
    fullResult.setAttribute("class", "none");
    result.setAttribute("class", "none");
    latencyDiv.setAttribute("class", "latency none");

    videoElement = $("#video");
    canvasElement = $("#canvas");
    overlayElement = $("#overlay");
    statusElement = $("#status");
    fpsElement = $("#fps");
    confidenceSlider = $("#confidence");
    confidenceValue = $("#confidence-value");

    controls();
    updateUi();
    showCompatibleChromiumVersion("image-classification");
    const transformersJs = $("#ortversion");
    transformersJs.innerHTML = `<a href="https://huggingface.co/docs/transformers.js/en/index">Transformer.js</a>`;

    console.log(`${provider} ${deviceType} ${modelName}`);

    start.addEventListener(
        "click",
        async () => {
            start.disabled = true;
            stop.disabled = false;
            await startDetection();
        },
        false,
    );

    stop.addEventListener(
        "click",
        () => {
            start.disabled = false;
            stop.disabled = true;
            stopDetection();
        },
        false,
    );
};

document.addEventListener("DOMContentLoaded", ui, false);
