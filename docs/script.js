
const MODEL_PATH = 'https://huggingface.co/voyagerfromeast/skyseg/resolve/main/skyseg_fp16.onnx';
const INPUT_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const CACHE_NAME = 'sky-seg-model-v1';

// Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const originalCanvas = document.getElementById('original-canvas');
const maskCanvas = document.getElementById('mask-canvas');
const resetBtn = document.getElementById('reset-btn');
const downloadBtn = document.getElementById('download-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');

let session = null;

// Initialize
async function init() {
    try {
        loadingText.innerText = "Checking cache for model...";
        progressContainer.classList.remove('hidden');

        const modelBuffer = await fetchAndCacheModel(MODEL_PATH);

        loadingText.innerText = "Initializing session...";
        // Create an inference session
        session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ['wasm'] });

        loadingOverlay.classList.add('hidden');
        console.log("Model loaded successfully");
    } catch (e) {
        console.error("Failed to load model:", e);
        loadingText.innerText = "Error loading model. Check console.\n" + e.message;
        loadingText.style.color = "red";
        progressContainer.classList.add('hidden');
    }
}

async function fetchAndCacheModel(url) {
    try {
        if (!window.caches) {
            throw new Error("Cache API not supported");
        }

        const cache = await caches.open(CACHE_NAME);
        const cachedResponse = await cache.match(url);

        if (cachedResponse) {
            console.log("Loading model from cache...");
            loadingText.innerText = "Loading model from cache...";
            progressBar.style.width = '100%';
            return await cachedResponse.arrayBuffer();
        }

        console.log("Downloading model...");
        loadingText.innerText = "Downloading model (0%)...";

        const response = await fetchWithProgress(url);
        const responseToCache = response.clone();
        await cache.put(url, responseToCache);

        return await response.arrayBuffer();
    } catch (e) {
        console.error("Cache error:", e);
        throw e;
    }
}

async function fetchWithProgress(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const contentLength = response.headers.get('content-length');
    if (!contentLength) {
        return response;
    }

    const total = parseInt(contentLength, 10);
    let loaded = 0;

    const reader = response.body.getReader();
    const stream = new ReadableStream({
        async start(controller) {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                loaded += value.byteLength;
                const progress = (loaded / total) * 100;

                requestAnimationFrame(() => {
                    progressBar.style.width = `${progress}%`;
                    loadingText.innerText = `Downloading model (${Math.round(progress)}%)...`;
                });

                controller.enqueue(value);
            }
            controller.close();
        }
    });

    return new Response(stream, {
        headers: response.headers
    });
}

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        processImage(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        processImage(e.target.files[0]);
    }
});

resetBtn.addEventListener('click', () => {
    previewArea.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';

    const ctx1 = originalCanvas.getContext('2d');
    const ctx2 = maskCanvas.getContext('2d');
    ctx1.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
    ctx2.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
});

downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'mask.png';
    link.href = maskCanvas.toDataURL();
    link.click();
});

async function processImage(file) {
    if (!session) {
        alert("Model not loaded yet or failed to load.");
        return;
    }

    // UI Updates
    dropZone.classList.add('hidden');
    previewArea.classList.remove('hidden');
    loadingOverlay.classList.remove('hidden');
    loadingText.innerText = "Processing...";

    try {
        const image = await loadImage(file);
        const inputTensor = preprocess(image);

        // Run inference
        const feeds = {};
        const inputName = session.inputNames[0];
        feeds[inputName] = inputTensor;

        const results = await session.run(feeds);
        const outputName = session.outputNames[0];
        const outputMap = results[outputName];

        originalCanvas.width = image.width;
        originalCanvas.height = image.height;
        originalCanvas.getContext('2d').drawImage(image, 0, 0);

        drawMask(outputMap.data, image.width, image.height);

    } catch (e) {
        console.error("Inference failed:", e);
        alert("Inference failed. Check console.");
        previewArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
    } finally {
        loadingOverlay.classList.add('hidden');
        loadingText.innerText = "";
    }
}

function loadImage(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

function preprocess(image) {
    // Resize to INPUT_SIZE x INPUT_SIZE
    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = imageData;

    // Convert to Float32 and Normalize
    // Shape: [1, 3, 320, 320] -> NCHW
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        // Pixel indices
        const r = data[i * 4] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;

        // Standardize: (Value - Mean) / Std
        // R channel
        float32Data[i] = (r - MEAN[0]) / STD[0];
        // G channel (offset by 320*320)
        float32Data[i + INPUT_SIZE * INPUT_SIZE] = (g - MEAN[1]) / STD[1];
        // B channel (offset by 2*320*320)
        float32Data[i + 2 * INPUT_SIZE * INPUT_SIZE] = (b - MEAN[2]) / STD[2];
    }

    return new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function drawMask(data, originalWidth, originalHeight) {
    const cvs = document.createElement('canvas');
    cvs.width = INPUT_SIZE;
    cvs.height = INPUT_SIZE;
    const ctx = cvs.getContext('2d');

    const imageData = ctx.createImageData(INPUT_SIZE, INPUT_SIZE);
    const pixels = imageData.data;

    let min = 1000, max = -1000;

    for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        let val = data[i];

        if (val < min) min = val;
        if (val > max) max = val;

        const intensity = val * 255;

        pixels[i * 4] = intensity;     // R
        pixels[i * 4 + 1] = intensity; // G
        pixels[i * 4 + 2] = intensity; // B
        pixels[i * 4 + 3] = 255;       // Alpha
    }

    ctx.putImageData(imageData, 0, 0);

    maskCanvas.width = originalWidth;
    maskCanvas.height = originalHeight;
    const destCtx = maskCanvas.getContext('2d');
    destCtx.imageSmoothingEnabled = true;
    destCtx.imageSmoothingQuality = 'high';

    destCtx.drawImage(cvs, 0, 0, originalWidth, originalHeight);

    console.log(`Mask range: min=${min}, max=${max}`);
}

init();
