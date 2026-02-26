/**
 * DoodleAI - Neural network classifier for doodle recognition.
 * Uses WebGPU via webgpu-torch for GPU-accelerated training and inference.
 *
 * Architecture: 3-layer MLP
 *   Input  : 784  (28×28 grayscale, normalised, inverted)
 *   Hidden1: 128  (ReLU)
 *   Hidden2: 64   (ReLU)
 *   Output : N    (one neuron per class)
 */

const IMG_SIZE     = 28;   // resize drawings to 28×28 before feeding the net
const HIDDEN1_SIZE = 128;
const HIDDEN2_SIZE = 64;
const BATCH_SIZE   = 8;

class DoodleClassifier {
    constructor() {
        this.trainingData = [];   // Array of { pixels: number[], labelIdx: number }
        this.labelNames   = [];   // Ordered list of unique label strings
        this.W1 = null; this.b1 = null;
        this.W2 = null; this.b2 = null;
        this.W3 = null; this.b3 = null;
        this.modelTrained = false;
        this.gpuAvailable = false;
    }

    /** Initialise WebGPU.  Must be called before train() / predict(). */
    async init() {
        try {
            this.gpuAvailable = await torch.initWebGPUAsync();
        } catch (_) {
            this.gpuAvailable = false;
        }
        return this.gpuAvailable;
    }

    /**
     * Resize a canvas to 28×28 and convert to a normalised grayscale float
     * array with the background mapped to 0 and the drawn strokes to 1.
     */
    preprocessCanvas(canvas) {
        const tmp = document.createElement('canvas');
        tmp.width  = IMG_SIZE;
        tmp.height = IMG_SIZE;
        const ctx = tmp.getContext('2d');
        ctx.drawImage(canvas, 0, 0, IMG_SIZE, IMG_SIZE);
        const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
        const pixels = [];
        for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
            const gray = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3.0;
            pixels.push(1.0 - gray / 255.0);   // invert: white bg → 0, ink → 1
        }
        return pixels;
    }

    /**
     * Add a labelled drawing from the current canvas to the training set.
     * Returns a summary object with sample counts per label.
     */
    addSample(canvas, label) {
        const trimmed = label.trim().toLowerCase();
        if (!trimmed) throw new Error('Label cannot be empty');

        const pixels = this.preprocessCanvas(canvas);
        if (!this.labelNames.includes(trimmed)) {
            this.labelNames.push(trimmed);
        }
        const labelIdx = this.labelNames.indexOf(trimmed);
        this.trainingData.push({ pixels, labelIdx });
        this.modelTrained = false;  // requires re-training after new data

        return {
            totalSamples: this.trainingData.length,
            labels: [...this.labelNames],
            counts: this.labelNames.map(l => ({
                label: l,
                count: this.trainingData.filter(
                    d => d.labelIdx === this.labelNames.indexOf(l)
                ).length
            }))
        };
    }

    /** Fisher-Yates shuffle – unbiased in-place shuffle of a copy of the array. */
    _shuffle(arr) {
        const a = [...arr];
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }


    _randomMatrix(rows, cols) {
        const scale = Math.sqrt(2.0 / rows);
        const arr = [];
        for (let i = 0; i < rows; i++) {
            arr[i] = [];
            for (let j = 0; j < cols; j++) {
                arr[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
        return arr;
    }

    /** Create a float32 tensor whose gradients will be tracked. */
    _param(data) {
        const t = torch.tensor(data);
        t.requiresGrad = true;
        return t;
    }

    /** Allocate WebGPU tensors for all learnable parameters. */
    _initWeights() {
        const n         = this.labelNames.length;
        const inputSize = IMG_SIZE * IMG_SIZE;   // 784

        this.W1 = this._param(this._randomMatrix(inputSize,    HIDDEN1_SIZE));
        this.b1 = this._param([new Array(HIDDEN1_SIZE).fill(0.0)]);
        this.W2 = this._param(this._randomMatrix(HIDDEN1_SIZE, HIDDEN2_SIZE));
        this.b2 = this._param([new Array(HIDDEN2_SIZE).fill(0.0)]);
        this.W3 = this._param(this._randomMatrix(HIDDEN2_SIZE, n));
        this.b3 = this._param([new Array(n).fill(0.0)]);
    }

    /**
     * Forward pass through the 3-layer MLP.
     * x shape: [batch, 784]  →  returns logits [batch, numClasses]
     */
    _forward(x) {
        const h1 = torch.relu(torch.add(torch.mm(x,  this.W1), this.b1));  // [batch, 128]
        const h2 = torch.relu(torch.add(torch.mm(h1, this.W2), this.b2));  // [batch,  64]
        return torch.add(torch.mm(h2, this.W3), this.b3);                  // [batch,   N]
    }

    /**
     * Zero out the accumulated gradients on all learnable parameters.
     * (torch.optim is not in the public webgpu-torch API so we manage
     * gradients manually.)
     */
    _zeroGrads() {
        for (const p of [this.W1, this.b1, this.W2, this.b2, this.W3, this.b3]) {
            if (p) p.grad = null;
        }
    }

    /**
     * Manual SGD step: p ← p − lr * p.grad
     * Detaches each updated tensor so the next forward pass starts a fresh graph.
     */
    _sgdStep(lr) {
        const update = (p) => {
            if (!p || !p.grad) return p;
            const newVal = torch.sub(p, torch.mul(p.grad, lr));
            const leaf   = newVal.detach();   // create a new leaf tensor
            leaf.requiresGrad = true;
            return leaf;
        };
        this.W1 = update(this.W1);
        this.b1 = update(this.b1);
        this.W2 = update(this.W2);
        this.b2 = update(this.b2);
        this.W3 = update(this.W3);
        this.b3 = update(this.b3);
    }

    /**
     * Mean-squared-error loss between raw logits and one-hot targets.
     *
     * NOTE: Cross-entropy is the theoretically preferred loss for multi-class
     * classification, but webgpu-torch v0.3.5 does not export
     * `torch.nn.functional.crossEntropyLoss` (it exists in source but is not
     * in the public bundle API).  MSE with one-hot targets is used here as a
     * fully supported alternative — it still minimises correctly and backprops
     * through the same graph.
     *
     * @param {Tensor} logits – [batchSize, numClasses]
     * @param {Array}  batch  – array of { pixels, labelIdx }
     * @returns {Tensor} scalar loss
     */
    _computeLoss(logits, batch) {
        const numClasses = this.labelNames.length;
        const oneHotData = batch.map(d => {
            const row = new Array(numClasses).fill(0.0);
            row[d.labelIdx] = 1.0;
            return row;
        });
        const targets = torch.tensor(oneHotData);              // [batch, numClasses]
        const diff    = torch.sub(logits, targets);            // [batch, numClasses]
        return torch.mean(torch.mul(diff, diff));              // scalar
    }

    /**
     * Train the model using GPU-accelerated mini-batch SGD.
     *
     * @param {number}   epochs     – number of full passes over training data
     * @param {number}   lr         – SGD learning rate
     * @param {Function} onProgress – optional callback(epoch, total, avgLoss)
     */
    async train(epochs = 100, lr = 0.01, onProgress = null) {
        if (this.trainingData.length < 2) {
            throw new Error('Need at least 2 training samples');
        }
        if (this.labelNames.length < 2) {
            throw new Error('Need samples from at least 2 different labels');
        }

        this._initWeights();

        for (let epoch = 0; epoch < epochs; epoch++) {
            // Shuffle training data before each epoch using Fisher-Yates
            const shuffled   = this._shuffle(this.trainingData);
            let   totalLoss  = 0;
            let   batchCount = 0;
            const batchSize  = Math.min(BATCH_SIZE, shuffled.length);

            for (let i = 0; i < shuffled.length; i += batchSize) {
                const batch  = shuffled.slice(i, i + batchSize);
                const xBatch = torch.tensor(batch.map(d => d.pixels));

                this._zeroGrads();
                const logits = this._forward(xBatch);
                const loss   = this._computeLoss(logits, batch);
                loss.backward();
                this._sgdStep(lr);

                // toArrayAsync() is the correct API (tensor.item() is not exported)
                const lossVal = await loss.toArrayAsync();
                if (typeof lossVal === 'number') {
                    totalLoss += lossVal;
                } else if (Array.isArray(lossVal) && lossVal.length > 0) {
                    totalLoss += lossVal[0];
                } else {
                    console.warn('Unexpected lossVal format from toArrayAsync():', lossVal);
                }
                batchCount++;
            }

            if (onProgress) {
                onProgress(epoch + 1, epochs, totalLoss / batchCount);
            }
            // Yield to the browser event loop so the UI stays responsive
            await new Promise(r => setTimeout(r, 0));
        }

        this.modelTrained = true;
    }

    /**
     * Run inference on the current canvas drawing.
     * Returns labels sorted by confidence (highest first).
     */
    async predict(canvas) {
        if (!this.modelTrained) {
            throw new Error('Model not trained yet. Add samples and click Train Model.');
        }
        const pixels = this.preprocessCanvas(canvas);
        const x      = torch.tensor([pixels]);
        const logits = this._forward(x);

        // Manual softmax: exp(logits) / sum(exp(logits))
        // torch.softmax is not a standalone export in webgpu-torch v0.3.5
        const expLogits = torch.exp(logits);
        // toArrayAsync() on a [1, numClasses] tensor returns [[v0, v1, ...]]
        const expData   = await expLogits.toArrayAsync();
        const row       = Array.isArray(expData[0]) ? expData[0] : expData;
        if (!Array.isArray(row) || row.length !== this.labelNames.length) {
            throw new Error('Unexpected inference output shape from model');
        }
        const total = row.reduce((s, v) => s + v, 0);

        return this.labelNames
            .map((label, i) => ({ label, confidence: row[i] / total }))
            .sort((a, b) => b.confidence - a.confidence);
    }

    /** Remove all stored samples and reset model weights. */
    clearData() {
        this.trainingData = [];
        this.labelNames   = [];
        this.W1 = this.b1 = this.W2 = this.b2 = this.W3 = this.b3 = null;
        this.modelTrained = false;
    }

    getStatus() {
        return {
            gpuAvailable: this.gpuAvailable,
            modelTrained: this.modelTrained,
            sampleCount:  this.trainingData.length,
            labels:       [...this.labelNames]
        };
    }
}

// Singleton instance used by the page
const classifier = new DoodleClassifier();
