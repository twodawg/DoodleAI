function getImage() {
	const canvas = document.getElementById('scratchPad');
	//const ctx = canvas.getContext('2d');

	const dataUrl = canvas.toDataURL('image/png');

	return dataUrl;
}

// Tool state: 'draw' | 'erase'
let _currentTool = 'draw';

/** Switch between draw and erase modes. */
function setDrawTool(tool) {
	_currentTool = tool;
	const canvas = document.getElementById('scratchPad');
	canvas.style.cursor = (tool === 'erase') ? 'cell' : 'crosshair';
}

/**
 * Load a stored training sample (pixel float array) back onto the drawing
 * canvas so the user can edit it.
 *
 * @param {number[]} pixels  – normalised grayscale array (0 = white, 1 = black)
 * @param {number}   imgSize – width/height of the stored sample (e.g. 28, 56)
 */
function loadSampleToCanvas(pixels, imgSize) {
	const canvas = document.getElementById('scratchPad');
	const ctx    = canvas.getContext('2d', { willReadFrequently: true });

	// Build a small canvas at the stored resolution
	const tmp    = document.createElement('canvas');
	tmp.width    = imgSize;
	tmp.height   = imgSize;
	const tmpCtx = tmp.getContext('2d');
	const imgData = tmpCtx.createImageData(imgSize, imgSize);
	for (let i = 0; i < imgSize * imgSize; i++) {
		const val = Math.round((1.0 - pixels[i]) * 255);
		imgData.data[i * 4]     = val;
		imgData.data[i * 4 + 1] = val;
		imgData.data[i * 4 + 2] = val;
		imgData.data[i * 4 + 3] = 255;
	}
	tmpCtx.putImageData(imgData, 0, 0);

	// Clear the scratchPad to white, then scale the sample up to fill it
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	ctx.imageSmoothingEnabled = false;
	ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

function initDraw() {
	const canvas = document.getElementById('scratchPad');
	const toolbar = document.getElementById('toolbar');

	const ctx = canvas.getContext('2d', { willReadFrequently: true });

	//const canvasOffsetX = canvas.offsetLeft;
	//const canvasOffsetY = canvas.offsetTop;

	//canvas.width = window.innerWidth - canvasOffsetX;
	//canvas.height = window.innerHeight - canvasOffsetY;

	canvas.width = 224;
	canvas.height = 224;

	ctx.fillStyle = "white";
	ctx.fillRect(0, 0, canvas.width, canvas.height);

	let isPainting = false;
	let lineWidth = 2;

	const toolbarClick = (e) => {
		if (e.target.id === 'clear') {
			ctx.clearRect(0, 0, canvas.width, canvas.height);
			ctx.fillStyle = "white";
			ctx.fillRect(0, 0, canvas.width, canvas.height);
		}
	};

	toolbar.addEventListener('click', toolbarClick);
	toolbar.addEventListener('touchstart', toolbarClick);

	toolbar.addEventListener('change', e => {
		if (e.target.id === 'stroke') {
			ctx.strokeStyle = e.target.value;
		}

		if (e.target.id === 'lineWidth') {
			lineWidth = e.target.value;
		}
	});

	
	const canvasMouseDown = (e) => {
		isPainting = true;
		if (_currentTool === 'erase') {
			ctx.strokeStyle = 'white';
		} else {
			const color = document.getElementById('stroke');
			ctx.strokeStyle = color.value;
		}

		startX = e.clientX;
		startY = e.clientY;
	};

	canvas.addEventListener('mousedown', canvasMouseDown);
	canvas.addEventListener('touchstart', canvasMouseDown);
	
	const canvasMouseUp = (e) => {
		isPainting = false;
		ctx.stroke();
		ctx.beginPath();
	};
	
	canvas.addEventListener('mouseup', canvasMouseUp);
	canvas.addEventListener('touchend', canvasMouseUp);


	const drawFunc = (e) => {
		if (!isPainting) {
			return;
		}

		ctx.lineWidth = lineWidth;
		ctx.lineCap = 'round';
		if (_currentTool === 'erase') {
			ctx.strokeStyle = 'white';
		}

		//ctx.lineTo(e.clientX - canvasOffsetX, e.clientY);
		var mousePos = getMousePos(canvas, e);
		ctx.lineTo(mousePos.x, mousePos.y);

		ctx.stroke();

		// End the line if the mouse leaves the canvas bounds
		if (mousePos.x < 4 || mousePos.x > canvas.width - 4 ||
			mousePos.y < 4 || mousePos.y > canvas.height - 4) {
			isPainting = false;
			ctx.stroke();
			ctx.beginPath();
		}

	}
	function getMousePos(canvas, evt) {
		var rect = canvas.getBoundingClientRect();
		return {
			x: evt.clientX - rect.left,
			y: evt.clientY - rect.top
		};
	}
	function touchMove(e) { drawFunc(e.touches[0]); e.preventDefault(); };

	canvas.addEventListener('mousemove', drawFunc);
	canvas.addEventListener('touchmove', touchMove);
}