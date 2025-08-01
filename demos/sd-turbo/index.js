/* eslint-disable no-undef */
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run sd-turbo with webnn in onnxruntime-web.
//

import {
    $,
    $$,
    convertToFloat16OrUint16Array,
    convertToFloat32Array,
    log,
    logError,
    setupORT,
    showCompatibleChromiumVersion,
    toHalf,
} from "../../assets/js/common_utils.js";

/*
 * get configuration from url
 */
function getConfig() {
    const queryParams = new URLSearchParams(window.location.search);
    const config = {
        model: location.href.includes("github.io")
            ? "https://huggingface.co/microsoft/sd-turbo-webnn/resolve/main"
            : "models",
        mode: "none",
        safetyChecker: true,
        provider: "webnn",
        deviceType: "gpu",
        threads: 1,
        images: 4,
        ort: "test",
        logOutput: false,
        verbose: false,
    };

    for (const key in config) {
        const lowerKey = key.toLowerCase();
        const value = queryParams.get(key) ?? queryParams.get(lowerKey);
        if (value !== null) {
            if (typeof config[key] === "boolean") {
                config[key] = value === "true";
            } else if (typeof config[key] === "number") {
                config[key] = isNaN(parseInt(value)) ? config[key] : parseInt(value);
            } else {
                config[key] = decodeURIComponent(value);
            }
        }
    }

    return config;
}

/*
 * initialize latents with random noise
 */
function randn_latents(shape, noise_sigma) {
    function randn() {
        // Use the Box-Muller transform
        let u = Math.random();
        let v = Math.random();
        let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z;
    }
    let size = 1;
    shape.forEach(element => {
        size *= element;
    });

    let data = new Float32Array(size);
    // Loop over the shape dimensions
    for (let i = 0; i < size; i++) {
        data[i] = randn() * noise_sigma;
    }
    return data;
}

let device = "gpu";
let badge;
let memoryReleaseSwitch;
let textEncoderFetchProgress = 0;
let unetFetchProgress = 0;
let vaeDecoderFetchProgress = 0;
let textEncoderCompileProgress = 0;
let unetCompileProgress = 0;
let vaeDecoderCompileProgress = 0;
let scFetchProgress = 0;
let scCompileProgress = 0;

// Get model via Origin Private File System
async function getModelOPFS(name, url, updateModel) {
    const root = await navigator.storage.getDirectory();
    let fileHandle;

    async function updateFile() {
        const response = await fetch(url);
        const buffer = await readResponse(name, response);
        fileHandle = await root.getFileHandle(name, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(buffer);
        await writable.close();
        return buffer;
    }

    if (updateModel) {
        return await updateFile();
    }

    try {
        fileHandle = await root.getFileHandle(name);
        const blob = await fileHandle.getFile();
        let buffer = await blob.arrayBuffer();
        if (buffer) {
            if (getSafetyChecker()) {
                if (name == "sd_turbo_text_encoder") {
                    textEncoderFetchProgress = 20.0;
                } else if (name == "sd_turbo_unet") {
                    unetFetchProgress = 48.0;
                } else if (name == "sd_turbo_vae_decoder") {
                    vaeDecoderFetchProgress = 3.0;
                } else if (name == "sd_turbo_safety_checker") {
                    scFetchProgress = 17.0;
                }
            } else {
                if (name == "sd_turbo_text_encoder") {
                    textEncoderFetchProgress = 20.0;
                } else if (name == "sd_turbo_unet") {
                    unetFetchProgress = 65.0;
                } else if (name == "sd_turbo_vae_decoder") {
                    vaeDecoderFetchProgress = 3.0;
                }
            }

            updateProgress();
            updateLoadWave(progress.toFixed(2));
            return buffer;
        }
    } catch (e) {
        console.log(e.message);
        return await updateFile();
    }
}

async function readResponse(name, response) {
    const contentLength = response.headers.get("Content-Length");
    let total = parseInt(contentLength ?? "0");
    let buffer = new Uint8Array(total);
    let loaded = 0;

    const reader = response.body.getReader();
    async function read() {
        const { done, value } = await reader.read();
        if (done) return;

        let newLoaded = loaded + value.length;
        let fetchProgress = (newLoaded / contentLength) * 100;

        if (!getSafetyChecker()) {
            if (name == "sd_turbo_text_encoder") {
                textEncoderFetchProgress = 0.2 * fetchProgress;
            } else if (name == "sd_turbo_unet") {
                unetFetchProgress = 0.65 * fetchProgress;
            } else if (name == "sd_turbo_vae_decoder") {
                vaeDecoderFetchProgress = 0.03 * fetchProgress;
            }
        } else {
            if (name == "sd_turbo_text_encoder") {
                textEncoderFetchProgress = 0.2 * fetchProgress;
            } else if (name == "sd_turbo_unet") {
                unetFetchProgress = 0.48 * fetchProgress;
            } else if (name == "sd_turbo_vae_decoder") {
                vaeDecoderFetchProgress = 0.03 * fetchProgress;
            } else if (name == "sd_turbo_safety_checker") {
                scFetchProgress = 0.17 * fetchProgress;
            }
        }

        updateProgress();
        updateLoadWave(progress.toFixed(2));

        if (newLoaded > total) {
            total = newLoaded;
            let newBuffer = new Uint8Array(total);
            newBuffer.set(buffer);
            buffer = newBuffer;
        }
        buffer.set(value, loaded);
        loaded = newLoaded;
        return read();
    }

    await read();
    return buffer;
}

const getMode = () => {
    return getQueryValue("mode") === "normal" ? false : true;
};

const getSafetyChecker = () => {
    if (getQueryValue("safetychecker")) {
        return getQueryValue("safetychecker") === "true" ? true : false;
    } else {
        return true;
    }
};

const updateProgress = () => {
    progress =
        textEncoderFetchProgress +
        unetFetchProgress +
        scFetchProgress +
        vaeDecoderFetchProgress +
        textEncoderCompileProgress +
        unetCompileProgress +
        vaeDecoderCompileProgress +
        scCompileProgress;
};

/*
 * load models used in the pipeline
 */
async function load_models(models) {
    log("[Load] ONNX Runtime Execution Provider: " + config.provider);
    log("[Load] ONNX Runtime EP device type: " + config.deviceType);
    updateLoadWave(0.0);
    load.disabled = true;

    for (const [name, model] of Object.entries(models)) {
        let modelNameInLog = "";
        try {
            let start = performance.now();
            let modelUrl;
            if (name == "text_encoder") {
                modelNameInLog = "Text Encoder";
                modelUrl = `${config.model}/${name}/model_layernorm.onnx`;
            } else if (name == "unet") {
                modelNameInLog = "UNet";
                modelUrl = `${config.model}/${name}/model_layernorm.onnx`;
            } else if (name == "vae_decoder") {
                modelNameInLog = "VAE Decoder";
                modelUrl = `${config.model}/${name}/model.onnx`;
            } else if (name == "safety_checker") {
                modelNameInLog = "Safety Checker";
                modelUrl = `${config.model}/${name}/safety_checker_int32_reduceSum.onnx`;
            }
            log(`[Load] Loading model ${modelNameInLog} · ${model.size}`);
            let modelBuffer = await getModelOPFS(`sd_turbo_${name}`, modelUrl, false);
            let modelFetchTime = (performance.now() - start).toFixed(2);
            if (name == "text_encoder") {
                textEncoderFetch.innerHTML = modelFetchTime;
            } else if (name == "unet") {
                unetFetch.innerHTML = modelFetchTime;
            } else if (name == "vae_decoder") {
                vaeFetch.innerHTML = modelFetchTime;
            } else if (name == "safety_checker") {
                scFetch.innerHTML = modelFetchTime;
            }
            log(`[Load] ${modelNameInLog} loaded · ${modelFetchTime}ms`);
            log(`[Session Create] Beginning ${modelNameInLog}`);

            start = performance.now();
            const sess_opt = { ...opt, ...model.opt };
            console.log(sess_opt);
            models[name].sess = await ort.InferenceSession.create(modelBuffer, sess_opt);
            let createTime = (performance.now() - start).toFixed(2);

            if (getSafetyChecker()) {
                if (name == "text_encoder") {
                    textEncoderCreate.innerHTML = createTime;
                    textEncoderCompileProgress = 2;
                    updateProgress();
                    updateLoadWave(progress.toFixed(2));
                } else if (name == "unet") {
                    unetCreate.innerHTML = createTime;
                    unetCompileProgress = 7;
                    updateProgress();
                    updateLoadWave(progress.toFixed(2));
                } else if (name == "vae_decoder") {
                    vaeCreate.innerHTML = createTime;
                    vaeDecoderCompileProgress = 1;
                    updateProgress();
                    updateLoadWave(progress.toFixed(2));
                } else if (name == "safety_checker") {
                    scCreate.innerHTML = createTime;
                    scCompileProgress = 2;
                    updateProgress();
                    updateLoadWave(progress.toFixed(2));
                }
            } else {
                if (name == "text_encoder") {
                    textEncoderCreate.innerHTML = createTime;
                    textEncoderCompileProgress = 2;
                    updateProgress();
                    updateLoadWave(progress.toFixed(2));
                } else if (name == "unet") {
                    unetCreate.innerHTML = createTime;
                    unetCompileProgress = 9;
                    updateProgress();
                    updateLoadWave(progress.toFixed(2));
                } else if (name == "vae_decoder") {
                    vaeCreate.innerHTML = createTime;
                    vaeDecoderCompileProgress = 1;
                    updateProgress();
                    updateLoadWave(progress.toFixed(2));
                }
            }

            if (getMode()) {
                log(`[Session Create] Create ${modelNameInLog} completed · ${createTime}ms`);
            } else {
                log(`[Session Create] Create ${modelNameInLog} completed`);
            }
        } catch (e) {
            log(`[Load] ${modelNameInLog} failed, ${e}`);
        }
    }

    updateLoadWave(100.0);
    log("[Session Create] Ready to generate images");
    let image_area = $$("#image_area>div");
    image_area.forEach(i => {
        i.setAttribute("class", "frame ready");
    });
    buttons.setAttribute("class", "button-group key loaded");
    generate.disabled = false;
    $("#user-input").setAttribute("class", "form-control enabled");
}

const config = getConfig();

let models = {
    unet: {
        // original model from dw, then wm dump new one from local graph optimization.
        url: "unet/model_layernorm.onnx",
        size: "1.61GB",
        opt: { graphOptimizationLevel: "disabled" }, // avoid wasm heap issue (need Wasm memory 64)
    },
    text_encoder: {
        // orignal model from gu, wm convert the output to fp16.
        url: "text_encoder/model_layernorm.onnx",
        size: "649MB",
        opt: { graphOptimizationLevel: "disabled" },
        // opt: { freeDimensionOverrides: { batch_size: 1, sequence_length: 77 } },
    },
    vae_decoder: {
        // use gu's model has precision lose in webnn caused by instanceNorm op,
        // covert the model to run instanceNorm in fp32 (insert cast nodes).
        url: "vae_decoder/model.onnx",
        size: "94.5MB",
        // opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } }
        opt: {
            freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64 },
        },
    },
    safety_checker: {
        url: "safety_checker/model.onnx",
        size: "580MB",
        opt: {
            freeDimensionOverrides: {
                batch: 1,
                channels: 3,
                height: 224,
                width: 224,
            },
        },
    },
};

let progress = 0;

let tokenizer;
let loading;
const sigma = 14.6146;
const gamma = 0;
const vae_scaling_factor = 0.18215;

const opt = {
    executionProviders: [config.provider],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: {
        session: {
            disable_prepacking: "1",
            use_device_allocator_for_initializers: "1",
            use_ort_model_bytes_directly: "1",
            use_ort_model_bytes_for_initializers: "1",
        },
    },
    logSeverityLevel: config.verbose ? 0 : 3, // 0: verbose, 1: info, 2: warning, 3: error
};

/*
 * scale the latents
 */
function scale_model_inputs(t) {
    const d_i = t.data;
    const d_o = new Float32Array(d_i.length);

    const divi = (sigma ** 2 + 1) ** 0.5;
    for (let i = 0; i < d_i.length; i++) {
        d_o[i] = d_i[i] / divi;
    }
    return new ort.Tensor(d_o, t.dims);
}

/*
 * Poor mens EulerA step
 * Since this example is just sd-turbo, implement the absolute minimum needed to create an image
 * Maybe next step is to support all sd flavors and create a small helper model in onnx can deal
 * much more efficient with latents.
 */
function step(model_output, sample) {
    const d_o = new Float32Array(model_output.data.length);
    const prev_sample = new ort.Tensor(d_o, model_output.dims);
    const sigma_hat = sigma * (gamma + 1);

    for (let i = 0; i < model_output.data.length; i++) {
        const pred_original_sample = sample.data[i] - sigma_hat * model_output.data[i];
        const derivative = (sample.data[i] - pred_original_sample) / sigma_hat;
        const dt = 0 - sigma_hat;
        d_o[i] = (sample.data[i] + derivative * dt) / vae_scaling_factor;
    }
    return prev_sample;
}

// eslint-disable-next-line no-unused-vars
function draw_out_image(t) {
    const imageData = t.toImageData({ tensorLayout: "NHWC", format: "RGB" });
    const canvas = $(`#img_canvas_safety`);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext("2d").putImageData(imageData, 0, 0);
}

function resize_image(image_nr, targetWidth, targetHeight) {
    // let canvas = document.createElement('canvas');
    // Use img_canvas_test to ensure the input
    const canvas = $(`#img_canvas_test`);
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    let ctx = canvas.getContext("2d");
    let canvas_source = $(`#img_canvas_${image_nr}`);
    ctx.drawImage(canvas_source, 0, 0, canvas_source.width, canvas_source.height, 0, 0, targetWidth, targetHeight);
    let imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);

    return imageData;
}

function normalizeImageData(imageData) {
    const mean = [0.48145466, 0.4578275, 0.40821073];
    const std = [0.26862954, 0.26130258, 0.27577711];
    const { data, width, height } = imageData;
    const numPixels = width * height;

    let array = new Float32Array(numPixels * 4).fill(0);

    for (let i = 0; i < numPixels; i++) {
        const offset = i * 4;
        for (let c = 0; c < 3; c++) {
            const normalizedValue = (data[offset + c] / 255 - mean[c]) / std[c];
            // data[offset + c] = Math.round(normalizedValue * 255);
            array[offset + c] = normalizedValue * 255;
        }
    }

    // return imageData;
    return { data: array, width: width, height: height };
}

function get_tensor_from_image(imageData, format) {
    const { data, width, height } = imageData;
    const numPixels = width * height;
    const channels = 3;
    const rearrangedData = new Float32Array(numPixels * channels);
    let destOffset = 0;

    for (let i = 0; i < numPixels; i++) {
        const srcOffset = i * 4;
        const r = data[srcOffset] / 255;
        const g = data[srcOffset + 1] / 255;
        const b = data[srcOffset + 2] / 255;

        if (format === "NCHW") {
            rearrangedData[destOffset] = r;
            rearrangedData[destOffset + numPixels] = g;
            rearrangedData[destOffset + 2 * numPixels] = b;
            destOffset++;
        } else if (format === "NHWC") {
            rearrangedData[destOffset] = r;
            rearrangedData[destOffset + 1] = g;
            rearrangedData[destOffset + 2] = b;
            destOffset += channels;
        } else {
            throw new Error("Invalid format specified.");
        }
    }

    const tensorShape = format === "NCHW" ? [1, channels, height, width] : [1, height, width, channels];
    let tensor = new ort.Tensor("float16", convertToFloat16OrUint16Array(rearrangedData), tensorShape);

    return tensor;
}

/**
 * draw an image from tensor
 * @param {ort.Tensor} t
 * @param {number} image_nr
 */
function draw_image(t, image_nr) {
    let pix = t.data;
    for (var i = 0; i < pix.length; i++) {
        let x = pix[i];
        x = x / 2 + 0.5;
        if (x < 0) x = 0;
        if (x > 1) x = 1;
        pix[i] = x;
    }
    const imageData = t.toImageData({ tensorLayout: "NCHW", format: "RGB" });
    const canvas = $(`#img_canvas_${image_nr}`);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext("2d").putImageData(imageData, 0, 0);
}

async function generate_image() {
    const img_divs = [img_div_0, img_div_1, img_div_2, img_div_3];
    img_divs.forEach(div => div.setAttribute("class", "frame"));

    try {
        textEncoderRun1.innerHTML = "";
        textEncoderRun2.innerHTML = "";
        textEncoderRun3.innerHTML = "";
        textEncoderRun4.innerHTML = "";
        unetRun1.innerHTML = "";
        unetRun2.innerHTML = "";
        unetRun3.innerHTML = "";
        unetRun4.innerHTML = "";
        vaeRun1.innerHTML = "";
        vaeRun2.innerHTML = "";
        vaeRun3.innerHTML = "";
        vaeRun4.innerHTML = "";
        runTotal1.innerHTML = "";
        runTotal2.innerHTML = "";
        runTotal3.innerHTML = "";
        runTotal4.innerHTML = "";
        scRun1.innerHTML = "";
        scRun2.innerHTML = "";
        scRun3.innerHTML = "";
        scRun4.innerHTML = "";

        if (getMode()) {
            for (let i = 1; i <= config.images; i++) {
                $(`#data${i}`).innerHTML = "... ms";
                $(`#data${i}`).setAttribute("class", "show");
            }
        } else {
            for (let i = 1; i <= config.images; i++) {
                $(`#data${i}`).innerHTML = "...";
                $(`#data${i}`).setAttribute("class", "show");
            }
        }

        log(`[Session Run] Beginning`);

        await loading;

        const prompt = $("#user-input");
        const { input_ids } = await tokenizer(prompt.value, {
            padding: true,
            max_length: 77,
            truncation: true,
            return_tensor: false,
        });

        // text_encoder
        let start = performance.now();
        const { last_hidden_state } = await models.text_encoder.sess.run({
            input_ids: new ort.Tensor("int32", input_ids, [1, input_ids.length]),
        });
        if (config.logOutput) {
            console.log(`[Model Output] Text Encoder - last_hidden_state:`);
            console.log(last_hidden_state.data);
        }

        let sessionRunTimeTextEncode = (performance.now() - start).toFixed(2);
        textEncoderRun1.innerHTML = sessionRunTimeTextEncode;
        textEncoderRun2.innerHTML = sessionRunTimeTextEncode;
        textEncoderRun3.innerHTML = sessionRunTimeTextEncode;
        textEncoderRun4.innerHTML = sessionRunTimeTextEncode;

        if (getMode()) {
            log(`[Session Run] Text Encoder execution time: ${sessionRunTimeTextEncode}ms`);
        } else {
            log(`[Session Run] Text Encoder completed`);
        }

        for (let j = 0; j < config.images; j++) {
            $(`#img_div_${j}`).setAttribute("class", "frame inferncing");
            let startTotal = performance.now();
            const latent_shape = [1, 4, 64, 64];
            let latent = new ort.Tensor(randn_latents(latent_shape, sigma), latent_shape);
            const latent_model_input = scale_model_inputs(latent);

            // unet
            start = performance.now();
            let feed = {
                sample: new ort.Tensor(
                    "float16",
                    convertToFloat16OrUint16Array(latent_model_input.data),
                    latent_model_input.dims,
                ),
                timestep: new ort.Tensor("float16", new Uint16Array([toHalf(999)]), [1]),
                encoder_hidden_states: last_hidden_state,
            };
            let { out_sample } = await models.unet.sess.run(feed);
            if (config.logOutput) {
                console.log(`[Model Output][Image ${j + 1}] UNet - out_sample: `);
                console.log(last_hidden_state.data);
            }
            let unetRunTime = (performance.now() - start).toFixed(2);
            $(`#unetRun${j + 1}`).innerHTML = unetRunTime;

            if (getMode()) {
                log(`[Session Run][Image ${j + 1}] UNet execution time: ${unetRunTime}ms`);
            } else {
                log(`[Session Run][Image ${j + 1}] UNet completed`);
            }

            // scheduler
            const new_latents = step(
                new ort.Tensor("float32", convertToFloat32Array(out_sample.data), out_sample.dims),
                latent,
            );

            // vae_decoder
            start = performance.now();
            const { sample } = await models.vae_decoder.sess.run({
                latent_sample: new_latents,
            });
            if (config.logOutput) {
                console.log(`[Model Output][Image ${j + 1}] VAE Decoder - sample:`);
                console.log(sample.data);
            }
            let vaeRunTime = (performance.now() - start).toFixed(2);
            $(`#vaeRun${j + 1}`).innerHTML = vaeRunTime;

            if (getMode()) {
                log(`[Session Run][Image ${j + 1}] VAE Decoder execution time: ${vaeRunTime}ms`);
            } else {
                log(`[Session Run][Image ${j + 1}] VAE Decoder completed`);
            }

            draw_image(sample, j);

            let totalRunTime = (performance.now() + Number(sessionRunTimeTextEncode) - startTotal).toFixed(2);
            if (getMode()) {
                log(`[Total] Image ${j + 1} execution time: ${totalRunTime}ms`);
            }
            $(`#runTotal${j + 1}`).innerHTML = totalRunTime;
            $(`#data${j + 1}`).setAttribute("class", "show");

            if (getMode()) {
                $(`#data${j + 1}`).innerHTML = totalRunTime + "ms";
            } else {
                $(`#data${j + 1}`).innerHTML = `${j + 1}`;
            }

            if (getSafetyChecker()) {
                // safety_checker
                let resized_image_data = resize_image(j, 224, 224);
                let normalized_image_data = normalizeImageData(resized_image_data);
                let safety_checker_feed = {
                    clip_input: get_tensor_from_image(normalized_image_data, "NCHW"),
                    images: get_tensor_from_image(resized_image_data, "NHWC"),
                };
                start = performance.now();
                const { has_nsfw_concepts } = await models.safety_checker.sess.run(safety_checker_feed);
                if (config.logOutput) {
                    console.log(`[Model Output][Image ${j + 1}] Safety Checker - has_nsfw_concepts:`);
                    console.log(has_nsfw_concepts.data);
                }

                let scRunTime = (performance.now() - start).toFixed(2);
                $(`#scRun${j + 1}`).innerHTML = scRunTime;

                if (getMode()) {
                    log(`[Session Run][Image ${j + 1}] Safety Checker execution time: ${scRunTime}ms`);
                } else {
                    log(`[Session Run][Image ${j + 1}] Safety Checker completed`);
                }

                $(`#img_div_${j}`).setAttribute("class", "frame done");

                let nsfw = false;
                has_nsfw_concepts.data[0] ? (nsfw = true) : (nsfw = false);
                log(`[Session Run][Image ${j + 1}] Safety Checker - not safe for work (NSFW) concepts: ${nsfw}`);

                if (has_nsfw_concepts.data[0]) {
                    $(`#img_div_${j}`).setAttribute("class", "frame done nsfw");
                    if (getMode()) {
                        $(`#data${j + 1}`).innerHTML = totalRunTime + "ms · NSFW";
                    } else {
                        $(`#data${j + 1}`).innerHTML = `${j + 1} · NSFW`;
                    }
                    $(`#data${j + 1}`).setAttribute("class", "nsfw show");
                    $(`#data${j + 1}`).setAttribute("title", "Not safe for work (NSFW) content");
                } else {
                    $(`#img_div_${j}`).setAttribute("class", "frame done");
                }
            } else {
                $(`#img_div_${j}`).setAttribute("class", "frame done");
            }

            // let out_image = new ort.Tensor("float32", convertToFloat32Array(out_images.data), out_images.dims);
            // draw_out_image(out_image);
        }
        // this is a gpu-buffer we own, so we need to dispose it
        last_hidden_state.dispose();
        log("[Info] Images generation completed");
    } catch (e) {
        log("[Error] " + e);
    }
}

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter.features.has("shader-f16");
    } catch (e) {
        console.log(e.message);
        return false;
    }
}

let webnnStatus;

const checkWebNN = async () => {
    let status = $("#webnnstatus");
    let info = $("#info");
    webnnStatus = await getWebnnStatus();

    if (webnnStatus.webnn) {
        status.setAttribute("class", "green");
        info.innerHTML = "WebNN supported";
        updateDeviceTypeLinks();
        load.disabled = false;
    } else {
        if (webnnStatus.error) {
            status.setAttribute("class", "red");
            info.innerHTML = `WebNN not supported: ${webnnStatus.error} <a id="webnn_na" href="../../install.html" title="WebNN Installation Guide">Set up WebNN</a>`;
            logError(`[Error] ${webnnStatus.error}`);
        } else {
            status.setAttribute("class", "red");
            info.innerHTML = "WebNN not supported";
            logError(`[Error] WebNN not supported`);
        }
    }

    if (getQueryValue("provider") && getQueryValue("provider").toLowerCase() === "webgpu") {
        status.innerHTML = "";
    }
};

const getWebnnStatus = async () => {
    let result = {};
    try {
        const context = await navigator.ml.createContext();
        if (context) {
            try {
                const builder = new MLGraphBuilder(context);
                if (builder) {
                    result.webnn = true;
                    return result;
                } else {
                    result.webnn = false;
                    return result;
                }
            } catch (e) {
                result.webnn = false;
                result.error = e.message;
                return result;
            }
        } else {
            result.webnn = false;
            return result;
        }
    } catch (ex) {
        result.webnn = false;
        result.error = ex.message;
        return result;
    }
};

const getQueryValue = name => {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
};

let textEncoderFetch = null;
let textEncoderCreate = null;
let textEncoderRun1 = null;
let textEncoderRun2 = null;
let textEncoderRun3 = null;
let textEncoderRun4 = null;
let unetFetch = null;
let unetCreate = null;
let unetRun1 = null;
let unetRun2 = null;
let unetRun3 = null;
let unetRun4 = null;
let vaeRun1 = null;
let vaeRun2 = null;
let vaeRun3 = null;
let vaeRun4 = null;
let scFetch = null;
let scCreate = null;
let scRun1 = null;
let scRun2 = null;
let scRun3 = null;
let scRun4 = null;
let vaeFetch = null;
let vaeCreate = null;
let runTotal1 = null;
let runTotal2 = null;
let runTotal3 = null;
let runTotal4 = null;
let generate = null;
let load = null;
let buttons = null;
let loadwave = null;
let loadwaveData = null;

const updateLoadWave = value => {
    loadwave = $$(".loadwave");
    loadwaveData = $$(".loadwave-data strong");

    if (loadwave && loadwaveData) {
        loadwave.forEach(l => {
            l.style.setProperty(`--loadwave-value`, value);
        });
        loadwaveData.forEach(data => {
            data.innerHTML = value;
        });

        if (value === 100) {
            loadwave.forEach(l => {
                l.dataset.value = value;
            });
        }
    }
};

const updateDeviceTypeLinks = () => {
    let backendLinks = $("#backend-links");
    const links = `· <a href="./?devicetype=gpu">GPU</a> · <a id="npu_link" href="./?devicetype=npu">NPU</a>`;
    backendLinks.innerHTML = `${links}`;
};

const ui = async () => {
    memoryReleaseSwitch = $("#memory_release");
    device = $("#device");
    badge = $("#badge");
    const prompt = $("#user-input");
    const title = $("#title");
    const dev = $("#dev");
    const dataElement = $("#data");
    const scTr = $("#scTr");
    load = $("#load");
    generate = $("#generate");
    buttons = $("#buttons");

    memoryReleaseSwitch.addEventListener("change", () => {
        if (memoryReleaseSwitch.checked) {
            memoryReleaseSwitch.setAttribute("checked", "");
        } else {
            memoryReleaseSwitch.removeAttribute("checked");
        }
    });

    if (!getMode()) {
        dev.setAttribute("class", "mt-1");
        dataElement.setAttribute("class", "hide");
    }

    await setupORT("sd-turbo", "dev");
    showCompatibleChromiumVersion("sd-turbo");

    if (getQueryValue("provider") && getQueryValue("provider").toLowerCase() === "webgpu") {
        title.innerHTML = "WebGPU";
    }
    await checkWebNN();

    const elementIds = [
        "#textEncoderFetch",
        "#textEncoderCreate",
        "#textEncoderRun1",
        "#textEncoderRun2",
        "#textEncoderRun3",
        "#textEncoderRun4",
        "#unetRun1",
        "#unetRun2",
        "#unetRun3",
        "#unetRun4",
        "#runTotal1",
        "#runTotal2",
        "#runTotal3",
        "#runTotal4",
        "#unetFetch",
        "#unetCreate",
        "#vaeFetch",
        "#vaeCreate",
        "#vaeRun1",
        "#vaeRun2",
        "#vaeRun3",
        "#vaeRun4",
        "#scFetch",
        "#scCreate",
        "#scRun1",
        "#scRun2",
        "#scRun3",
        "#scRun4",
    ];

    [
        textEncoderFetch,
        textEncoderCreate,
        textEncoderRun1,
        textEncoderRun2,
        textEncoderRun3,
        textEncoderRun4,
        unetRun1,
        unetRun2,
        unetRun3,
        unetRun4,
        runTotal1,
        runTotal2,
        runTotal3,
        runTotal4,
        unetFetch,
        unetCreate,
        vaeFetch,
        vaeCreate,
        vaeRun1,
        vaeRun2,
        vaeRun3,
        vaeRun4,
        scFetch,
        scCreate,
        scRun1,
        scRun2,
        scRun3,
        scRun4,
    ] = elementIds.map(id => $(id));

    switch (config.provider) {
        case "webgpu":
            if (!("gpu" in navigator)) {
                throw new Error("webgpu is NOT supported");
            }
            opt.preferredOutputLocation = { last_hidden_state: "gpu-buffer" };
            break;
        case "webnn":
            webnnStatus = await getWebnnStatus();
            if (webnnStatus.webnn) {
                opt.executionProviders = [
                    {
                        name: "webnn",
                        deviceType: config.deviceType,
                    },
                ];
            }
            break;
    }

    const deviceType = config.deviceType.toLowerCase();
    const provider = config.provider.toLowerCase();

    if (deviceType === "cpu" || provider === "wasm") {
        device.innerHTML = "CPU";
        badge.setAttribute("class", "cpu");
        document.body.setAttribute("class", "cpu");
    } else if (deviceType === "gpu" || provider === "webgpu") {
        device.innerHTML = "GPU";
        badge.setAttribute("class", "");
        document.body.setAttribute("class", "gpu");
    } else if (deviceType === "npu") {
        device.innerHTML = "NPU";
        badge.setAttribute("class", "npu");
        document.body.setAttribute("class", "npu");
    }

    prompt.value =
        "a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo";
    // Event listener for Ctrl + Enter or CMD + Enter
    prompt.addEventListener("keydown", function (e) {
        if (e.ctrlKey && e.key === "Enter") {
            generate_image();
        }
    });
    generate.addEventListener("click", function () {
        generate_image();
    });

    const load_model_ui = () => {
        loading = load_models(models);
        const img_divs = [img_div_0, img_div_1, img_div_2, img_div_3];
        img_divs.forEach(div => div.setAttribute("class", "frame loadwave"));
        buttons.setAttribute("class", "button-group key loading");
    };

    load.addEventListener("click", () => {
        if (config.provider === "webgpu") {
            hasFp16().then(fp16 => {
                if (fp16) {
                    load_model_ui();
                } else {
                    log(`[Error] Your GPU or Browser doesn't support webgpu/f16`);
                }
            });
        } else {
            load_model_ui();
        }
    });

    // ort.env.wasm.wasmPaths = 'dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    let path = "";
    if (
        location.href.toLowerCase().indexOf("github.io") > -1 ||
        location.href.toLowerCase().indexOf("huggingface.co") > -1 ||
        location.href.toLowerCase().indexOf("vercel.app") > -1
    ) {
        path = "microsoft/sd-turbo-webnn/resolve/main/tokenizer";
    } else {
        path = "../../demos/sd-turbo/models/tokenizer";
    }

    tokenizer = await AutoTokenizer.from_pretrained(path);
    tokenizer.pad_token_id = 0;

    if (getSafetyChecker()) {
        scTr.setAttribute("class", "");
    } else {
        scTr.setAttribute("class", "hide");
    }

    if (!getSafetyChecker()) {
        models = {
            unet: {
                // original model from dw, then wm dump new one from local graph optimization.
                url: "unet/model_layernorm.onnx",
                size: "1.61GB",
                opt: { graphOptimizationLevel: "disabled" }, // avoid wasm heap issue (need Wasm memory 64)
            },
            text_encoder: {
                // orignal model from gu, wm convert the output to fp16.
                url: "text_encoder/model_layernorm.onnx",
                size: "649MB",
                opt: { graphOptimizationLevel: "disabled" },
                // opt: { freeDimensionOverrides: { batch_size: 1, sequence_length: 77 } },
            },
            vae_decoder: {
                // use gu's model has precision lose in webnn caused by instanceNorm op,
                // covert the model to run instanceNorm in fp32 (insert cast nodes).
                url: "vae_decoder/model.onnx",
                size: "94.5MB",
                // opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } }
                opt: {
                    freeDimensionOverrides: {
                        batch: 1,
                        channels: 4,
                        height: 64,
                        width: 64,
                    },
                },
            },
        };
    }

    window.addEventListener("beforeunload", () => {
        if (memoryReleaseSwitch.checked) {
            const sessions = [
                models.safety_checker.sess,
                models.vae_decoder.sess,
                models.unet.sess,
                models.text_encoder.sess,
            ];

            Promise.allSettled(sessions.filter(session => session).map(session => session.release())).catch(error =>
                console.error("Session release error:", error),
            );

            load.disabled = false;
            buttons.setAttribute("class", "button-group key");
            generate.disabled = true;
            $("#user-input").setAttribute("class", "form-control");
            updateLoadWave(0.0);
            const img_divs = [img_div_0, img_div_1, img_div_2, img_div_3];
            img_divs.forEach(div => div.setAttribute("class", "frame"));
            textEncoderFetchProgress = 0;
            unetFetchProgress = 0;
            vaeDecoderFetchProgress = 0;
            textEncoderCompileProgress = 0;
            unetCompileProgress = 0;
            vaeDecoderCompileProgress = 0;
            scFetchProgress = 0;
            scCompileProgress = 0;
            for (let i = 1; i <= config.images; i++) {
                $(`#data${i}`).innerHTML = "... ms";
                $(`#data${i}`).setAttribute("class", "");
            }
            textEncoderFetch.innerHTML = "";
            textEncoderCreate.innerHTML = "";
            textEncoderRun1.innerHTML = "";
            textEncoderRun2.innerHTML = "";
            textEncoderRun3.innerHTML = "";
            textEncoderRun4.innerHTML = "";
            unetFetch.innerHTML = "";
            unetCreate.innerHTML = "";
            unetRun1.innerHTML = "";
            unetRun2.innerHTML = "";
            unetRun3.innerHTML = "";
            unetRun4.innerHTML = "";
            vaeRun1.innerHTML = "";
            vaeRun2.innerHTML = "";
            vaeRun3.innerHTML = "";
            vaeRun4.innerHTML = "";
            scFetch.innerHTML = "";
            scCreate.innerHTML = "";
            scRun1.innerHTML = "";
            scRun2.innerHTML = "";
            scRun3.innerHTML = "";
            scRun4.innerHTML = "";
            vaeFetch.innerHTML = "";
            vaeCreate.innerHTML = "";
            runTotal1.innerHTML = "";
            runTotal2.innerHTML = "";
            runTotal3.innerHTML = "";
            runTotal4.innerHTML = "";
        }
    });
};

document.addEventListener("DOMContentLoaded", ui, false);
