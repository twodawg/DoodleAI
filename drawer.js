function getImage() {
	const canvas = document.getElementById('scratchPad');
	//const ctx = canvas.getContext('2d');

	const dataUrl = canvas.toDataURL('image/png');

	return dataUrl;
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

	toolbar.addEventListener('click', e => {
		if (e.target.id === 'clear') {
			ctx.clearRect(0, 0, canvas.width, canvas.height);
			ctx.fillStyle = "white";
			ctx.fillRect(0, 0, canvas.width, canvas.height);
		}
	});

	toolbar.addEventListener('change', e => {
		if (e.target.id === 'stroke') {
			ctx.strokeStyle = e.target.value;
		}

		if (e.target.id === 'lineWidth') {
			lineWidth = e.target.value;
		}
	});

	canvas.addEventListener('mousedown', (e) => {
		isPainting = true;
		const color = document.getElementById('stroke');
		ctx.strokeStyle = color.value;

		startX = e.clientX;
		startY = e.clientY;
	});

	canvas.addEventListener('mouseup', e => {
		isPainting = false;
		ctx.stroke();
		ctx.beginPath();
	});


	const drawFunc = (e) => {
		if (!isPainting) {
			return;
		}

		ctx.lineWidth = lineWidth;
		ctx.lineCap = 'round';

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

	canvas.addEventListener('mousemove', drawFunc);
}