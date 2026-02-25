// model.js - Simple feedforward neural network for in-browser doodle classification

const DOODLE_INPUT_SIZE = 28 * 28; // 28x28 grayscale pixels
const DOODLE_HIDDEN_1 = 128;
const DOODLE_HIDDEN_2 = 64;

// Global classifier instance used by index.html
const doodleClassifier = new DoodleClassifier();

// ─── High-level classifier ────────────────────────────────────────────────────

function DoodleClassifier() {
    this.classes = [];       // ordered list of unique label strings
    this.trainingData = [];  // [{input: Float32Array, classIdx: number, label: string}]
    this.model = null;
    this.isTrained = false;
}

DoodleClassifier.prototype.addSample = function (canvas, label) {
    label = label.trim().toLowerCase();
    if (!label) return false;

    if (!this.classes.includes(label)) {
        this.classes.push(label);
    }

    const input = preprocessCanvas(canvas);
    const classIdx = this.classes.indexOf(label);
    this.trainingData.push({ input, classIdx, label });
    this.isTrained = false;
    return true;
};

DoodleClassifier.prototype.getStats = function () {
    const stats = {};
    for (const { label } of this.trainingData) {
        stats[label] = (stats[label] || 0) + 1;
    }
    return stats;
};

DoodleClassifier.prototype.train = async function (onProgress) {
    if (this.classes.length < 2) {
        throw new Error('Need at least 2 different labels to train.');
    }
    if (this.trainingData.length < 2) {
        throw new Error('Need at least 2 samples to train.');
    }

    this.model = new NeuralNetwork(
        DOODLE_INPUT_SIZE, DOODLE_HIDDEN_1, DOODLE_HIDDEN_2, this.classes.length
    );

    const inputs = this.trainingData.map(d => d.input);
    const labels = this.trainingData.map(d => d.classIdx);
    const losses = await this.model.trainAsync(inputs, labels, 0.01, 50, onProgress);
    this.isTrained = true;
    return losses;
};

DoodleClassifier.prototype.predict = function (canvas) {
    if (!this.isTrained || !this.model) return null;
    const input = preprocessCanvas(canvas);
    return this.model.predictWithConfidence(input, this.classes);
};

// ─── Canvas preprocessing ─────────────────────────────────────────────────────

function preprocessCanvas(canvas) {
    const off = document.createElement('canvas');
    off.width = 28;
    off.height = 28;
    const ctx = off.getContext('2d');
    ctx.drawImage(canvas, 0, 0, 28, 28);

    const idata = ctx.getImageData(0, 0, 28, 28);
    const rgba = idata.data;
    const pixels = new Float32Array(28 * 28);

    for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
        // Greyscale + invert so white background = 0, drawn pixels = positive
        pixels[p] = 1.0 - (rgba[i] * 0.299 + rgba[i + 1] * 0.587 + rgba[i + 2] * 0.114) / 255.0;
    }
    return pixels;
}

// ─── Neural network ───────────────────────────────────────────────────────────

function NeuralNetwork(inputSize, hidden1, hidden2, outputSize) {
    this.inputSize = inputSize;
    this.hidden1 = hidden1;
    this.hidden2 = hidden2;
    this.outputSize = outputSize;
    this._initWeights();
}

NeuralNetwork.prototype._initWeights = function () {
    this.W1 = randomMatrix(this.hidden1,  this.inputSize, Math.sqrt(2.0 / this.inputSize));
    this.b1 = new Float32Array(this.hidden1);
    this.W2 = randomMatrix(this.hidden2,  this.hidden1,   Math.sqrt(2.0 / this.hidden1));
    this.b2 = new Float32Array(this.hidden2);
    this.W3 = randomMatrix(this.outputSize, this.hidden2, Math.sqrt(2.0 / this.hidden2));
    this.b3 = new Float32Array(this.outputSize);
};

NeuralNetwork.prototype.forward = function (x) {
    const z1 = matVec(this.W1, x, this.hidden1,     this.inputSize);
    for (let i = 0; i < this.hidden1; i++) z1[i] += this.b1[i];
    const a1 = relu(z1);

    const z2 = matVec(this.W2, a1, this.hidden2,    this.hidden1);
    for (let i = 0; i < this.hidden2; i++) z2[i] += this.b2[i];
    const a2 = relu(z2);

    const z3 = matVec(this.W3, a2, this.outputSize, this.hidden2);
    for (let i = 0; i < this.outputSize; i++) z3[i] += this.b3[i];
    const a3 = softmax(z3, this.outputSize);

    return { z1, a1, z2, a2, z3, a3 };
};

NeuralNetwork.prototype._backprop = function (x, target, lr) {
    const { z1, a1, z2, a2, z3, a3 } = this.forward(x);

    // Output gradient (softmax + cross-entropy combined)
    const dz3 = new Float32Array(a3);
    dz3[target] -= 1;

    // W3, b3
    for (let i = 0; i < this.outputSize; i++) {
        const g = dz3[i];
        for (let j = 0; j < this.hidden2; j++) this.W3[i][j] -= lr * g * a2[j];
        this.b3[i] -= lr * g;
    }

    // Backprop through hidden layer 2
    const da2 = new Float32Array(this.hidden2);
    for (let j = 0; j < this.hidden2; j++) {
        let s = 0;
        for (let i = 0; i < this.outputSize; i++) s += this.W3[i][j] * dz3[i];
        da2[j] = s;
    }
    const dz2 = reluGrad(da2, z2, this.hidden2);

    for (let i = 0; i < this.hidden2; i++) {
        const g = dz2[i];
        for (let j = 0; j < this.hidden1; j++) this.W2[i][j] -= lr * g * a1[j];
        this.b2[i] -= lr * g;
    }

    // Backprop through hidden layer 1
    const da1 = new Float32Array(this.hidden1);
    for (let j = 0; j < this.hidden1; j++) {
        let s = 0;
        for (let i = 0; i < this.hidden2; i++) s += this.W2[i][j] * dz2[i];
        da1[j] = s;
    }
    const dz1 = reluGrad(da1, z1, this.hidden1);

    for (let i = 0; i < this.hidden1; i++) {
        const g = dz1[i];
        for (let j = 0; j < this.inputSize; j++) this.W1[i][j] -= lr * g * x[j];
        this.b1[i] -= lr * g;
    }

    return -Math.log(a3[target] + 1e-10);
};

NeuralNetwork.prototype.trainAsync = async function (data, labels, lr, epochs, onProgress) {
    const n = data.length;
    const losses = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
        // Shuffle indices
        const idx = Array.from({ length: n }, (_, i) => i);
        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [idx[i], idx[j]] = [idx[j], idx[i]];
        }

        let totalLoss = 0;
        for (const i of idx) totalLoss += this._backprop(data[i], labels[i], lr);
        losses.push(totalLoss / n);

        if (onProgress) onProgress(epoch + 1, epochs, totalLoss / n);
        await new Promise(resolve => setTimeout(resolve, 0)); // yield to UI
    }
    return losses;
};

NeuralNetwork.prototype.predictWithConfidence = function (x, classes) {
    const { a3 } = this.forward(x);
    let maxIdx = 0;
    for (let i = 1; i < this.outputSize; i++) {
        if (a3[i] > a3[maxIdx]) maxIdx = i;
    }
    return { label: classes[maxIdx], confidence: Math.round(a3[maxIdx] * 100) };
};

// ─── Math helpers ─────────────────────────────────────────────────────────────

function randomMatrix(rows, cols, scale) {
    const mat = new Array(rows);
    for (let i = 0; i < rows; i++) {
        mat[i] = new Float32Array(cols);
        for (let j = 0; j < cols; j++) {
            mat[i][j] = (Math.random() * 2 - 1) * scale;
        }
    }
    return mat;
}

function matVec(W, x, rows, cols) {
    const out = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
        let s = 0;
        const Wi = W[i];
        for (let j = 0; j < cols; j++) s += Wi[j] * x[j];
        out[i] = s;
    }
    return out;
}

function relu(z) {
    const out = new Float32Array(z.length);
    for (let i = 0; i < z.length; i++) out[i] = z[i] > 0 ? z[i] : 0;
    return out;
}

function reluGrad(upstream, z, n) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = z[i] > 0 ? upstream[i] : 0;
    return out;
}

function softmax(z, n) {
    let maxZ = z[0];
    for (let i = 1; i < n; i++) if (z[i] > maxZ) maxZ = z[i];
    const out = new Float32Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) { out[i] = Math.exp(z[i] - maxZ); sum += out[i]; }
    for (let i = 0; i < n; i++) out[i] /= sum;
    return out;
}
