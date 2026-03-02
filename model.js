/**
 * DoodleAI - Neural network classifier for doodle recognition.
 * Implements a 3-layer MLP trained with mini-batch SGD in plain JavaScript.
 * WebGPU (via webgpu-torch) is checked for availability but computation runs
 * on the CPU so that training works across all browsers.
 *
 * Architecture:
 *   Input  : imgSize×imgSize (configurable grayscale, normalised, inverted)
 *   Hidden1: configurable (ReLU)
 *   Hidden2: configurable (ReLU)
 *   Output : N    (one neuron per class)
 *
 * All weight matrices are stored as row-major Float32Arrays.
 */

const IMG_SIZE   = 28;   // default image size
const BATCH_SIZE = 8;

/** Supported image sizes for preprocessing. */
const IMG_SIZE_OPTIONS = [14, 28, 56, 112];

/** Preset model architectures available for selection. */
const MODEL_CONFIGS = {
    tiny:   { hidden1:  64, hidden2:  32, label: 'Tiny (64→32)'     },
    small:  { hidden1: 128, hidden2:  64, label: 'Small (128→64)'   },
    medium: { hidden1: 256, hidden2: 128, label: 'Medium (256→128)' },
    large:  { hidden1: 512, hidden2: 256, label: 'Large (512→256)'  },
};

class DoodleClassifier {
    constructor() {
        this.trainingData = [];   // Array of { pixels: number[], labelIdx: number }
        this.labelNames   = [];   // Ordered list of unique label strings
        this.W1 = null; this.b1 = null;
        this.W2 = null; this.b2 = null;
        this.W3 = null; this.b3 = null;
        this.modelTrained = false;
        this.gpuAvailable = false;
        // Default to the 'small' architecture (original behaviour)
        this.hidden1Size  = MODEL_CONFIGS.small.hidden1;
        this.hidden2Size  = MODEL_CONFIGS.small.hidden2;
        this.imgSize      = IMG_SIZE;   // default 28×28
    }

    /**
     * Switch the model architecture.  Must be called before train().
     * @param {string} key – one of the keys in MODEL_CONFIGS (e.g. 'small')
     */
    setModelConfig(key) {
        const cfg = MODEL_CONFIGS[key];
        if (!cfg) throw new Error('Unknown model config: ' + key);
        this.hidden1Size  = cfg.hidden1;
        this.hidden2Size  = cfg.hidden2;
        this.modelTrained = false;
    }

    /**
     * Set the image size used for preprocessing.  Must be called before
     * adding training samples so that all pixel arrays have a consistent length.
     * @param {number} size – one of the values in IMG_SIZE_OPTIONS (e.g. 28)
     */
    setImgSize(size) {
        if (!IMG_SIZE_OPTIONS.includes(size)) {
            throw new Error('Unsupported image size: ' + size +
                '. Supported sizes: ' + IMG_SIZE_OPTIONS.join(', '));
        }
        this.imgSize      = size;
        this.modelTrained = false;
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
     * Resize a canvas to imgSize×imgSize and convert to a normalised grayscale
     * float array with the background mapped to 0 and the drawn strokes to 1.
     */
    preprocessCanvas(canvas) {
        const sz  = this.imgSize;
        const tmp = document.createElement('canvas');
        tmp.width  = sz;
        tmp.height = sz;
        const ctx = tmp.getContext('2d');
        ctx.drawImage(canvas, 0, 0, sz, sz);
        const { data } = ctx.getImageData(0, 0, sz, sz);
        const pixels = [];
        for (let i = 0; i < sz * sz; i++) {
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
        const arr   = new Float32Array(rows * cols);
        for (let i = 0; i < arr.length; i++) {
            arr[i] = (Math.random() * 2 - 1) * scale;
        }
        return arr;
    }

    // ── Pure-JS linear algebra helpers ──────────────────────────────────────

    /** A @ B  —  A: [m, k], B: [k, n] → C: [m, n] */
    _matmul(A, B, m, k, n) {
        const C = new Float32Array(m * n);
        for (let i = 0; i < m; i++) {
            const rowA = i * k;
            const rowC = i * n;
            for (let p = 0; p < k; p++) {
                const a    = A[rowA + p];
                const colB = p * n;
                for (let j = 0; j < n; j++) C[rowC + j] += a * B[colB + j];
            }
        }
        return C;
    }

    /** A^T @ B  —  A: [m, k], B: [m, n] → C: [k, n] */
    _matmulTA(A, B, m, k, n) {
        const C = new Float32Array(k * n);
        for (let i = 0; i < m; i++) {
            for (let p = 0; p < k; p++) {
                const a    = A[i * k + p];
                const rowC = p * n;
                for (let j = 0; j < n; j++) C[rowC + j] += a * B[i * n + j];
            }
        }
        return C;
    }

    /** A @ B^T  —  A: [m, k], B: [n, k] → C: [m, n] */
    _matmulTB(A, B, m, k, n) {
        const C = new Float32Array(m * n);
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let p = 0; p < k; p++) sum += A[i * k + p] * B[j * k + p];
                C[i * n + j] = sum;
            }
        }
        return C;
    }

    /** Y[i, j] = X[i, j] + b[j] */
    _addBias(X, b, rows, cols) {
        const Y = new Float32Array(rows * cols);
        for (let i = 0; i < rows; i++) {
            const off = i * cols;
            for (let j = 0; j < cols; j++) Y[off + j] = X[off + j] + b[j];
        }
        return Y;
    }

    /** Element-wise ReLU, returns a new Float32Array */
    _relu(X) {
        const Y = new Float32Array(X.length);
        for (let i = 0; i < X.length; i++) Y[i] = X[i] > 0 ? X[i] : 0;
        return Y;
    }

    // ── Weight initialisation ────────────────────────────────────────────────

    /** Allocate He-initialised weight matrices and zero bias vectors. */
    _initWeights() {
        const n         = this.labelNames.length;
        const inputSize = this.imgSize * this.imgSize;
        const h1Size    = this.hidden1Size;
        const h2Size    = this.hidden2Size;

        this.W1 = this._randomMatrix(inputSize, h1Size);  // [784 × h1]
        this.b1 = new Float32Array(h1Size);
        this.W2 = this._randomMatrix(h1Size, h2Size);     // [h1 × h2]
        this.b2 = new Float32Array(h2Size);
        this.W3 = this._randomMatrix(h2Size, n);          // [h2 × N]
        this.b3 = new Float32Array(n);
    }

    // ── Forward pass ────────────────────────────────────────────────────────

    /**
     * Forward pass through the 3-layer MLP.
     * x: Float32Array of length batchSize × 784
     * Returns an object containing all layer outputs and pre-activations
     * required for the backward pass.
     */
    _forward(x, batchSize) {
        const n         = this.labelNames.length;
        const inputSize = this.imgSize * this.imgSize;
        const h1Size    = this.hidden1Size;
        const h2Size    = this.hidden2Size;

        const h1Pre  = this._addBias(this._matmul(x,   this.W1, batchSize, inputSize, h1Size), this.b1, batchSize, h1Size);
        const h1     = this._relu(h1Pre);
        const h2Pre  = this._addBias(this._matmul(h1,  this.W2, batchSize, h1Size,    h2Size), this.b2, batchSize, h2Size);
        const h2     = this._relu(h2Pre);
        const logits = this._addBias(this._matmul(h2,  this.W3, batchSize, h2Size,    n),      this.b3, batchSize, n);

        return { x, h1, h1Pre, h2, h2Pre, logits };
    }

    /**
     * Mean-squared-error loss between raw logits and one-hot targets.
     * Returns a plain number (scalar) rather than a tensor.
     *
     * @param {Float32Array} logits  – [batchSize × numClasses]
     * @param {Float32Array} targets – one-hot [batchSize × numClasses]
     * @returns {number} scalar MSE loss
     */
    _computeLoss(logits, targets) {
        let sum = 0;
        for (let i = 0; i < logits.length; i++) {
            const d = logits[i] - targets[i];
            sum += d * d;
        }
        return sum / logits.length;
    }

    // ── Backward pass + SGD update ───────────────────────────────────────────

    /** Sum columns of a [rows × cols] matrix, returning a [cols] bias-gradient vector. */
    _sumCols(M, rows, cols) {
        const out = new Float32Array(cols);
        for (let j = 0; j < cols; j++) for (let i = 0; i < rows; i++) out[j] += M[i * cols + j];
        return out;
    }

    /**
     * Backpropagate MSE loss and apply one SGD step to all weight matrices.
     *
     * @param {object}       fwd      – result of _forward()
     * @param {Float32Array} targets  – one-hot labels [batchSize × numClasses]
     * @param {number}       batchSize
     * @param {number}       lr       – learning rate
     */
    _backward(fwd, targets, batchSize, lr) {
        const n      = this.labelNames.length;
        const bs     = batchSize;
        const h1Size = this.hidden1Size;
        const h2Size = this.hidden2Size;
        const { x, h1, h1Pre, h2, h2Pre, logits } = fwd;

        // dL/dlogits = 2 * (logits − targets) / (bs * n)
        const scale = 2.0 / (bs * n);
        const dOut  = new Float32Array(bs * n);
        for (let i = 0; i < dOut.length; i++) dOut[i] = scale * (logits[i] - targets[i]);

        // ── Layer 3 ──────────────────────────────────────────────────────────
        const dW3 = this._matmulTA(h2, dOut, bs, h2Size, n);           // [h2, N]
        const db3 = this._sumCols(dOut, bs, n);

        const dh2 = this._matmulTB(dOut, this.W3, bs, n, h2Size);      // [bs, h2]

        // ReLU backward through h2
        const dh2Pre = new Float32Array(bs * h2Size);
        for (let i = 0; i < dh2Pre.length; i++) dh2Pre[i] = h2Pre[i] > 0 ? dh2[i] : 0;

        // ── Layer 2 ──────────────────────────────────────────────────────────
        const dW2 = this._matmulTA(h1, dh2Pre, bs, h1Size, h2Size);    // [h1, h2]
        const db2 = this._sumCols(dh2Pre, bs, h2Size);

        const dh1 = this._matmulTB(dh2Pre, this.W2, bs, h2Size, h1Size); // [bs, h1]

        // ReLU backward through h1
        const dh1Pre = new Float32Array(bs * h1Size);
        for (let i = 0; i < dh1Pre.length; i++) dh1Pre[i] = h1Pre[i] > 0 ? dh1[i] : 0;

        // ── Layer 1 ──────────────────────────────────────────────────────────
        const dW1 = this._matmulTA(x, dh1Pre, bs, this.imgSize * this.imgSize, h1Size); // [inputSize, h1]
        const db1 = this._sumCols(dh1Pre, bs, h1Size);

        // ── SGD update ────────────────────────────────────────────────────────
        for (let i = 0; i < this.W1.length; i++) this.W1[i] -= lr * dW1[i];
        for (let i = 0; i < this.b1.length; i++) this.b1[i] -= lr * db1[i];
        for (let i = 0; i < this.W2.length; i++) this.W2[i] -= lr * dW2[i];
        for (let i = 0; i < this.b2.length; i++) this.b2[i] -= lr * db2[i];
        for (let i = 0; i < this.W3.length; i++) this.W3[i] -= lr * dW3[i];
        for (let i = 0; i < this.b3.length; i++) this.b3[i] -= lr * db3[i];
    }

    /**
     * Train the model using mini-batch SGD.
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
        const n = this.labelNames.length;

        for (let epoch = 0; epoch < epochs; epoch++) {
            // Shuffle training data before each epoch using Fisher-Yates
            const shuffled   = this._shuffle(this.trainingData);
            let   totalLoss  = 0;
            let   batchCount = 0;
            const batchSize  = Math.min(BATCH_SIZE, shuffled.length);

            for (let i = 0; i < shuffled.length; i += batchSize) {
                const batch = shuffled.slice(i, i + batchSize);
                const bs    = batch.length;

                // Build flat input and one-hot target arrays for this mini-batch
                const x       = new Float32Array(bs * this.imgSize * this.imgSize);
                const targets = new Float32Array(bs * n);
                for (let b = 0; b < bs; b++) {
                    x.set(batch[b].pixels, b * this.imgSize * this.imgSize);
                    targets[b * n + batch[b].labelIdx] = 1.0;
                }

                const fwd = this._forward(x, bs);
                totalLoss += this._computeLoss(fwd.logits, targets);
                this._backward(fwd, targets, bs, lr);
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
        const x      = new Float32Array(pixels);
        const { logits } = this._forward(x, 1);

        // Numerically stable softmax: shift by max before exp to avoid overflow
        let maxLogit = logits[0];
        for (let i = 1; i < logits.length; i++) if (logits[i] > maxLogit) maxLogit = logits[i];

        const exps  = new Float32Array(logits.length);
        let   total = 0;
        for (let i = 0; i < logits.length; i++) {
            exps[i] = Math.exp(logits[i] - maxLogit);
            total  += exps[i];
        }

        return this.labelNames
            .map((label, i) => ({ label, confidence: exps[i] / total }))
            .sort((a, b) => b.confidence - a.confidence);
    }

    /**
     * Export training data as a plain JSON-serialisable object.
     * The returned value can be passed to importTrainingData() to restore state.
     *
     * imgSize is derived from the actual pixel count of the stored samples so
     * that the exported value is always correct, even when setImgSize() was
     * called after samples had already been captured at a different resolution.
     */
    exportTrainingData() {
        let imgSize = this.imgSize;
        if (this.trainingData.length > 0) {
            const pixelLen = this.trainingData[0].pixels.length;
            const inferred = Math.round(Math.sqrt(pixelLen));
            if (inferred * inferred === pixelLen) imgSize = inferred;
        }
        return {
            imgSize,
            labelNames: [...this.labelNames],
            trainingData: this.trainingData.map(d => ({
                pixels: Array.from(d.pixels),
                labelIdx: d.labelIdx
            }))
        };
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /** Remove labels that have no remaining samples; renumber labelIdx values. */
    _pruneUnusedLabels() {
        const usedIndices = new Set(this.trainingData.map(d => d.labelIdx));
        if (usedIndices.size === this.labelNames.length) return;
        const remap = new Map();
        let   next  = 0;
        this.labelNames = this.labelNames.filter((_, oldIdx) => {
            if (!usedIndices.has(oldIdx)) return false;
            remap.set(oldIdx, next++);
            return true;
        });
        this.trainingData.forEach(d => { d.labelIdx = remap.get(d.labelIdx); });
    }

    /** Returns the standard training-status summary object. */
    _summary() {
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

    /**
     * Remove a single training sample by its index.
     * Labels that no longer have any samples are pruned automatically.
     * Returns the same summary object as addSample() for convenience.
     *
     * @param {number} index – zero-based position in trainingData
     */
    removeSample(index) {
        if (index < 0 || index >= this.trainingData.length) {
            throw new Error('Sample index out of bounds: ' + index);
        }
        this.trainingData.splice(index, 1);
        this._pruneUnusedLabels();
        this.modelTrained = false;
        return this._summary();
    }

    /**
     * Change the label of a training sample by its index.
     * If newLabel does not yet exist it is added to labelNames.
     * Labels that have no remaining samples are pruned afterwards.
     * Returns the same summary object as addSample() for convenience.
     *
     * @param {number} index    – zero-based position in trainingData
     * @param {string} newLabel – the desired label (trimmed, lowercased)
     */
    relabelSample(index, newLabel) {
        const trimmed = newLabel.trim().toLowerCase();
        if (!trimmed) throw new Error('Label cannot be empty');
        if (index < 0 || index >= this.trainingData.length) {
            throw new Error('Sample index out of bounds: ' + index);
        }
        if (!this.labelNames.includes(trimmed)) {
            this.labelNames.push(trimmed);
        }
        this.trainingData[index].labelIdx = this.labelNames.indexOf(trimmed);
        this._pruneUnusedLabels();
        this.modelTrained = false;
        return this._summary();
    }

    /**
     * Import training data from a previously exported object.
     * Returns the same summary object as addSample() for convenience.
     *
     * @param {object} data – { labelNames: string[], trainingData: { pixels: number[], labelIdx: number }[] }
     */
    importTrainingData(data) {
        if (!data || !Array.isArray(data.labelNames) || !Array.isArray(data.trainingData)) {
            throw new Error('Invalid training data format');
        }
        // imgSize is optional for backward-compatibility; default to 28 if absent
        let imgSize = (data.imgSize !== undefined) ? data.imgSize : IMG_SIZE;
        // If the declared imgSize doesn't match the actual pixel count of the
        // first sample, try to infer the true imgSize from the pixel data.
        if (data.trainingData.length > 0 && Array.isArray(data.trainingData[0].pixels)) {
            const actualCount   = data.trainingData[0].pixels.length;
            const inferredSize  = Math.round(Math.sqrt(actualCount));
            if (inferredSize * inferredSize === actualCount &&
                    IMG_SIZE_OPTIONS.includes(inferredSize) &&
                    inferredSize !== imgSize) {
                imgSize = inferredSize;
            }
        }
        if (!IMG_SIZE_OPTIONS.includes(imgSize)) {
            throw new Error('Unsupported imgSize in training data: ' + imgSize);
        }
        const numLabels = data.labelNames.length;
        const pixelLen  = imgSize * imgSize;

        this.labelNames   = [...data.labelNames];
        this.trainingData = data.trainingData.map((d, i) => {
            if (!Array.isArray(d.pixels) || d.pixels.length !== pixelLen) {
                throw new Error('Sample ' + i + ': pixels must be an array of ' + pixelLen + ' numbers');
            }
            if (!d.pixels.every(v => typeof v === 'number' && v >= 0 && v <= 1)) {
                throw new Error('Sample ' + i + ': pixel values must be numbers in the range [0, 1]');
            }
            if (!Number.isInteger(d.labelIdx) || d.labelIdx < 0 || d.labelIdx >= numLabels) {
                throw new Error('Sample ' + i + ': labelIdx ' + d.labelIdx + ' is out of bounds');
            }
            return { pixels: d.pixels, labelIdx: d.labelIdx };
        });
        this.imgSize      = imgSize;
        this.modelTrained = false;

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
