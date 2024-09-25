// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run Image Classification with webnn in onnxruntime-web.
//
import * as transformers from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.16';
import {
  log,
  logError,
  getQueryValue,
  webNnStatus,
  updateQueryStringParameter,
  getMedian,
  getAverage,
  getMinimum,
  asyncErrorHandling,
  getMode,
  showCompatibleChromiumVersion
} from "../../assets/js/common_utils.js";

transformers.env.backends.onnx.wasm.proxy = false;
transformers.env.backends.onnx.wasm.simd = true;
transformers.env.backends.onnx.wasm.numThreads = 1;

const useRemoteModels = location.hostname.includes('github.io');
transformers.env.allowRemoteModels = useRemoteModels;
transformers.env.allowLocalModels = !useRemoteModels;
log('[Transformer.js] env.allowRemoteModels: ' + transformers.env.allowRemoteModels);
log('[Transformer.js] env.allowLocalModels: ' + transformers.env.allowLocalModels);
if (transformers.env.allowLocalModels) {
  transformers.env.localModelPath = "./models/";
  log('[Transformer.js] env.localModelPath: ' + transformers.env.localModelPath);
}

let provider = "webnn";
let deviceType = "gpu";
let dataType = "fp16";
let modelId = "mobilenetv4s";
let modelPath = "onnx-community/mobilenetv4s-webnn";
let runs = 1;
let range, rangeValue, runSpan;
let backendLabels, modelLabels;
let label_webgpu,
  label_webnn_cpu,
  label_webnn_gpu,
  label_webnn_npu,
  label_mobilenetV2,
  label_mobilenetv4s;
let uploadImage, label_uploadImage;
let imageUrl, image;
let classify;
let fullResult;
let first, average, median, best, throughput;
let status, circle, info;

let result;
let label1, label2, label3;
let score1, score2, score3;
let dataTypeSpan;
let modelIdSpan;
let latency, latencyDiv;

const main = async () => {
  // xenova/mobilenetv4s
  // webnn/mobilenet-v2
  // webnn/squeezenet-1.0

  fullResult.setAttribute("class", "none");
  result.setAttribute("class", "none");
  latencyDiv.setAttribute("class", "latency none");
  label_uploadImage.setAttribute("class", "disabled");
  uploadImage.disabled = true;
  classify.disabled = true;

  let options = {
    device: provider,
    dtype: dataType,
  };

  if (getQueryValue("model")) {
    modelId = getQueryValue("model");
    switch (modelId) {
      case "mobilenet-v2":
        modelPath = "webnn/mobilenet-v2";
        break;
      case "mobilenetv4s":
        modelPath = "onnx-community/mobilenetv4s-webnn";
        break;
      default:
        modelPath = "onnx-community/mobilenetv4s-webnn";
        break;
      // webnn/squeezenet-1.0
    }
  }

  modelIdSpan.innerHTML = dataType;
  dataTypeSpan.innerHTML = modelPath;

  try {
    log("[ONNX Runtime] Options: " + JSON.stringify(options));
    log(
      `[Transformer.js] Loading ${modelPath} and running image-classification pipeline`
    );

    const classifier = await transformers.pipeline('image-classification', modelPath, options);

    let [err, output] = await asyncErrorHandling(
      classifier(imageUrl, { topk: 3 })
    );

    if (err) {
      status.setAttribute("class", "red");
      info.innerHTML = err.message;
      logError(err.message);
    } else {
      // if (getMode()) {
      //   log(JSON.stringify(transformers.getPerf()));
      //   let warmUp = transformers.getPerf().warmup;
      //   let averageInference = getAverage(transformers.getPerf().inference);
      //   let medianInference = getMedian(transformers.getPerf().inference);
      //   latency.innerHTML = medianInference.toFixed(2);
      //   first.innerHTML = warmUp.toFixed(2);
      //   average.innerHTML = averageInference;
      //   median.innerHTML = medianInference.toFixed(2);
      //   best.innerHTML = getMinimum(transformers.getPerf().inference);
      //   throughput.innerHTML = `${transformers.getPerf().throughput} FPS`;
      //   fullResult.setAttribute("class", "");
      //   latencyDiv.setAttribute("class", "latency");
      // }

      label1.innerHTML = output[0].label;
      score1.innerText = output[0].score;
      label2.innerText = output[1].label;
      score2.innerText = output[1].score;
      label3.innerText = output[2].label;
      score3.innerText = output[2].score;
      result.setAttribute("class", "");
      label_uploadImage.setAttribute("class", "");
      uploadImage.disabled = false;
      classify.disabled = false;
      log(JSON.stringify(output));
      log(`[Transformer.js] Classifier completed`);
    }
  } catch (err) {
    log(`[Error] ${err}`);

    status.setAttribute("class", "red");
    info.innerHTML = `
            ${error}<br>
            Your device probably doesn't have a supported GPU.`;
    label_uploadImage.setAttribute("class", "disabled");
    uploadImage.disabled = true;
    classify.disabled = true;
    log(`[Error] ${error}`);
    log(`[Error] Your device probably doesn't have a supported GPU`);
  }
};

const checkWebNN = async () => {
  let webnnStatus = await webNnStatus();

  if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("webgpu") > -1
  ) {
    circle.setAttribute("class", "none");
    info.innerHTML = "";
  }

  if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("webnn") > -1
  ) {
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
          label_uploadImage.setAttribute("class", "disabled");
          uploadImage.disabled = true;
          classify.disabled = true;
          log(`[Error] ${error}`);
          log(
            `[Error] Your device probably doesn't have an AI processor (NPU) or the NPU driver is not successfully installed`
          );
        }
      } else {
        label_uploadImage.setAttribute("class", "");
        uploadImage.disabled = false;
        classify.disabled = false;
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
      label_uploadImage.setAttribute("class", "disabled");
      uploadImage.disabled = true;
      classify.disabled = true;
    }
  }
};

const initModelSelector = () => {
  if (getQueryValue("provider") && getQueryValue("devicetype")) {
    backendLabels.forEach((label) => {
      label.setAttribute("class", "btn");
    });
    if (
      getQueryValue("provider").toLowerCase() === "webgpu" &&
      getQueryValue("devicetype").toLowerCase() === "gpu"
    ) {
      label_webgpu.setAttribute("class", "btn active");
    } else if (
      getQueryValue("provider").toLowerCase() === "webnn" &&
      getQueryValue("devicetype").toLowerCase() === "cpu"
    ) {
      label_webnn_cpu.setAttribute("class", "btn active");
    } else if (
      getQueryValue("provider").toLowerCase() === "webnn" &&
      getQueryValue("devicetype").toLowerCase() === "gpu"
    ) {
      label_webnn_gpu.setAttribute("class", "btn active");
    } else if (
      getQueryValue("provider").toLowerCase() === "webnn" &&
      getQueryValue("devicetype").toLowerCase() === "npu"
    ) {
      label_webnn_npu.setAttribute("class", "btn active");
    }
  }

  if (getQueryValue("model")) {
    modelLabels.forEach((label) => {
      label.setAttribute("class", "btn");
    });
    if (getQueryValue("model").toLowerCase() === "mobilenet-v2") {
      label_mobilenetV2.setAttribute("class", "btn active");
    } else if (getQueryValue("model").toLowerCase() === "mobilenetv4s") {
      label_mobilenetv4s.setAttribute("class", "btn active");
    }
  }
};

const controls = async () => {
  const getRange = () => {
    rangeValue.innerHTML = range.value;
    runs = parseInt(range.value);
    if (runs > 1) {
      runSpan.innerHTML = "runs";
    } else {
      runSpan.innerHTML = "run";
    }
    let currentUrl = window.location.href;
    let updatedUrl = updateQueryStringParameter(currentUrl, "run", runs);
    window.history.pushState({}, "", updatedUrl);
  };

  range.addEventListener("input", getRange, false);

  let backendBtns = document.querySelector("#backendBtns");
  let modelBtns = document.querySelector("#modelBtns");

  const updateBackend = (e) => {
    backendLabels.forEach((label) => {
      label.setAttribute("class", "btn");
    });
    e.target.parentNode.setAttribute("class", "btn active");
    if (e.target.id.trim() === "webgpu") {
      let currentUrl = window.location.href;
      let updatedUrl = updateQueryStringParameter(
        currentUrl,
        "provider",
        "webgpu"
      );
      window.history.pushState({}, "", updatedUrl);
      currentUrl = window.location.href;
      updatedUrl = updateQueryStringParameter(currentUrl, "devicetype", "gpu");
      window.history.pushState({}, "", updatedUrl);
      provider = "webgpu";
      deviceType = "gpu";
    } else if (e.target.id.trim() === "webnn_cpu") {
      let currentUrl = window.location.href;
      let updatedUrl = updateQueryStringParameter(
        currentUrl,
        "provider",
        "webnn"
      );
      window.history.pushState({}, "", updatedUrl);
      currentUrl = window.location.href;
      updatedUrl = updateQueryStringParameter(currentUrl, "devicetype", "cpu");
      window.history.pushState({}, "", updatedUrl);
      provider = "webnn";
      deviceType = "cpu";
    } else if (e.target.id.trim() === "webnn_gpu") {
      let currentUrl = window.location.href;
      let updatedUrl = updateQueryStringParameter(
        currentUrl,
        "provider",
        "webnn"
      );
      window.history.pushState({}, "", updatedUrl);
      currentUrl = window.location.href;
      updatedUrl = updateQueryStringParameter(currentUrl, "devicetype", "gpu");
      window.history.pushState({}, "", updatedUrl);
      provider = "webnn";
      deviceType = "gpu";
    } else if (e.target.id.trim() === "webnn_npu") {
      let currentUrl = window.location.href;
      let updatedUrl = updateQueryStringParameter(
        currentUrl,
        "provider",
        "webnn"
      );
      window.history.pushState({}, "", updatedUrl);
      currentUrl = window.location.href;
      updatedUrl = updateQueryStringParameter(currentUrl, "devicetype", "npu");
      window.history.pushState({}, "", updatedUrl);
      provider = "webgpu";
      deviceType = "npu";
    }

    updateUi();
  };

  const updateModel = (e) => {
    modelLabels.forEach((label) => {
      label.setAttribute("class", "btn");
    });
    e.target.parentNode.setAttribute("class", "btn active");
    if (e.target.id.trim() === "mobilenet-v2") {
      let currentUrl = window.location.href;
      let updatedUrl = updateQueryStringParameter(
        currentUrl,
        "model",
        "mobilenet-v2"
      );
      window.history.pushState({}, "", updatedUrl);
      modelId = "mobilenet-v2";
    } else if (e.target.id.trim() === "mobilenetv4s") {
      let currentUrl = window.location.href;
      let updatedUrl = updateQueryStringParameter(
        currentUrl,
        "model",
        "mobilenetv4s"
      );
      window.history.pushState({}, "", updatedUrl);
      modelId = "mobilenetv4s";
    }
    updateUi();
  };

  backendBtns.addEventListener("change", updateBackend, false);
  modelBtns.addEventListener("change", updateModel, false);
};

const badgeUpdate = () => {
  if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("webgpu") > -1
  ) {
    title.innerHTML = "WebGPU";
    provider = "webgpu";
    deviceType = "gpu";
    device.innerHTML = "GPU";
    badge.setAttribute("class", "gpu");
  } else if (
    getQueryValue("provider") &&
    getQueryValue("provider").toLowerCase().indexOf("wasm") > -1
  ) {
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
    if (
      getQueryValue("devicetype") &&
      getQueryValue("devicetype").toLowerCase() === "cpu"
    ) {
      deviceType = "cpu";
      device.innerHTML = "CPU";
      badge.setAttribute("class", "cpu");
    } else if (
      getQueryValue("devicetype") &&
      getQueryValue("devicetype").toLowerCase() === "npu"
    ) {
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
  if (getQueryValue("run")) {
    runs = parseInt(getQueryValue("run"));
    range.value = runs;
    rangeValue.innerHTML = runs;
  } else {
    runs = 1;
    range.value = 1;
    rangeValue.innerHTML = "1";
  }

  if (runs > 1) {
    runSpan.innerHTML = "runs";
  } else {
    runSpan.innerHTML = "run";
  }

  if (getQueryValue("model")) {
    modelId = getQueryValue("model");
  }

  initModelSelector();
  badgeUpdate();
  log(
    `[Config] Demo config updated · ${modelId} · ${provider} · ${deviceType}`
  );
  await checkWebNN();
  console.log(provider);
  console.log(deviceType);
  console.log(modelId);
  console.log(runs);
};

const changeImage = async () => {
  let file = uploadImage.files[0];
  if (file) {
    image.src = URL.createObjectURL(file);
    imageUrl = URL.createObjectURL(file);
  }
  await main();
};

const ui = async () => {
  imageUrl = "../image-classification/static/tiger.jpg";
  if (
    !(
      getQueryValue("provider") &&
      getQueryValue("model") &&
      getQueryValue("devicetype") &&
      getQueryValue("run")
    )
  ) {
    let url = "?provider=webnn&devicetype=gpu&model=mobilenetv4s&run=5";
    location.replace(url);
  }

  status = document.querySelector("#webnnstatus");
  circle = document.querySelector("#circle");
  info = document.querySelector("#info");
  range = document.querySelector("#range");
  rangeValue = document.querySelector(".rangevalue");
  runSpan = document.querySelector("#run-span");
  backendLabels = document.querySelectorAll(".backends label");
  modelLabels = document.querySelectorAll(".models label");
  label_webgpu = document.querySelector("#label_webgpu");
  label_webnn_cpu = document.querySelector("#label_webnn_cpu");
  label_webnn_gpu = document.querySelector("#label_webnn_gpu");
  label_webnn_npu = document.querySelector("#label_webnn_npu");
  label_mobilenetV2 = document.querySelector("#label_mobilenet-v2");
  label_mobilenetv4s = document.querySelector("#label_mobilenetv4s");
  image = document.querySelector("#image");
  uploadImage = document.querySelector("#upload-image");
  label_uploadImage = document.querySelector("#label_upload-image");
  classify = document.querySelector("#classify-image");
  fullResult = document.querySelector("#full-result");
  first = document.querySelector("#first");
  average = document.querySelector("#average");
  median = document.querySelector("#median");
  best = document.querySelector("#best");
  throughput = document.querySelector("#throughput");
  label1 = document.querySelector("#label1");
  label2 = document.querySelector("#label2");
  label3 = document.querySelector("#label3");
  score1 = document.querySelector("#score1");
  score2 = document.querySelector("#score2");
  score3 = document.querySelector("#score3");
  result = document.querySelector("#result");
  modelIdSpan = document.querySelector("#data-type");
  dataTypeSpan = document.querySelector("#model-id");
  latency = document.querySelector("#latency");
  latencyDiv = document.querySelector(".latency");
  fullResult.setAttribute("class", "none");
  result.setAttribute("class", "none");
  latencyDiv.setAttribute("class", "latency none");
  controls();
  updateUi();
  showCompatibleChromiumVersion('image-classification');
  const transformersjsversion = document.querySelector("#transformersjsversion");
  let transformersJsVersion = transformersjsversion.innerHTML;
  transformersjsversion.innerHTML = `<a href="https://huggingface.co/docs/transformers.js/en/index">Transformer.js · ${transformersJsVersion}</a>`;

  console.log(`${provider} ${deviceType} ${modelId} ${runs}`);

  classify.addEventListener(
    "click",
    async () => {
      await main();
    },
    false
  );

  uploadImage.addEventListener(
    "change",
    async () => {
      await changeImage();
    },
    false
  );
};

document.addEventListener("DOMContentLoaded", ui, false);