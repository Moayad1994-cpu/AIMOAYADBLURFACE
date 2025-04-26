// script.js (Combined Manual Draw + Haar Fallback - v8 Fixes)
document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileElem');
    const fileNameDisplay = document.getElementById('file-name-display');
    const drawPreviewSection = document.getElementById('draw-preview-section');
    const canvasContainer = document.getElementById('canvas-container');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const clearDrawingsBtn = document.getElementById('clear-drawings-btn');
    const regionCountDisplay = document.getElementById('region-count-display');
    const blurSlider = document.getElementById('blur-slider');
    const blurValueDisplay = document.getElementById('blur-value');
    const processBtn = document.getElementById('process-btn');
    const processingIndicator = document.getElementById('processing-indicator');
    const outputArea = document.getElementById('output-area');
    const resultActionsDiv = document.getElementById('result-actions'); // Changed from downloadArea
    const downloadLink = document.getElementById('download-link');
    const uploadNewBtn = document.getElementById('upload-new-btn');   // New button
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const body = document.body;
    const status_label = document.getElementById('status-label');

    // --- State Variables ---
    let currentMediaFile = null;
    let canvas = null;
    let ctx = null;
    let previewImage = null; // Source Image/Video for canvas drawing
    let isDrawing = false;
    let startX, startY;
    let drawnRegions = []; // Stores {x, y, w, h} relative to *original* media dimensions
    let scaleFactor = 1;
    let objectUrl = null; // For video/image preview URL management

    // --- Initial Setup ---
    if (!status_label) console.error("Status label element not found!"); // Check on load

    // --- Theme Toggle ---
    function applyTheme(theme) { /* ... (unchanged) ... */ body.classList.remove('light-mode', 'dark-mode'); body.classList.add(theme); themeToggleBtn.innerHTML = (theme === 'dark-mode') ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>'; localStorage.setItem('theme', theme); }
    const savedTheme = localStorage.getItem('theme') || 'light-mode'; applyTheme(savedTheme);
    themeToggleBtn.addEventListener('click', () => { const newTheme = body.classList.contains('dark-mode') ? 'light-mode' : 'dark-mode'; applyTheme(newTheme); });

    // --- Blur Slider ---
    blurSlider.addEventListener('input', () => { blurValueDisplay.textContent = parseFloat(blurSlider.value).toFixed(2); });

    // --- Drag and Drop & File Handling ---
    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => { dropArea.addEventListener(eventName, preventDefaults, false); document.body.addEventListener(eventName, preventDefaults, false); });
    ['dragenter', 'dragover'].forEach(eventName => { dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false); });
    ['dragleave', 'drop'].forEach(eventName => { dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false); });
    dropArea.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);
    function handleDrop(e) { const files = e.dataTransfer.files; if (files.length > 0) { handleMediaFile(files[0]); } }
    function handleFileSelect(e) { if (e.target.files.length > 0) { handleMediaFile(e.target.files[0]); } }

    function handleMediaFile(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'video/mp4', 'video/webm', 'video/ogg', 'video/avi', 'video/mov', 'video/mkv', 'video/wmv', 'video/flv'];
        let isAllowed = allowedTypes.includes(file.type) || file.type.startsWith('image/') || file.type.startsWith('video/');
        if (!isAllowed) { alert(`Unsupported file type: ${file.type || 'unknown'}.`); resetAllStates(); return; }
        const maxSizeMB = 100; const maxSize = maxSizeMB * 1024 * 1024;
        if (file.size > maxSize) { alert(`File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max size: ${maxSizeMB} MB.`); resetAllStates(); return; }

        resetAllStates(); // Reset everything before loading new file preview

        currentMediaFile = file;
        fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        setupCanvasForPreview(file); // Setup canvas for drawing preview
        outputArea.innerHTML = '<p>Processed output will appear here.</p>'; // Reset output placeholder
        resultActionsDiv.style.display = 'none'; // Hide download/upload new buttons
        processingIndicator.style.display = 'none';
        processBtn.disabled = false; // Enable process button
        if (status_label) status_label.textContent = "File ready. Draw regions or click Process.";
    }

    // --- Canvas Setup and Drawing ---
    function setupCanvasForPreview(file) {
        clearCanvasAndRegions(); // Clear previous drawings and revoke old Object URL
        previewPlaceholder.style.display = 'none';
        drawPreviewSection.style.display = 'block';

        // --- Use URL.createObjectURL for BOTH image and video ---
        if (objectUrl) { URL.revokeObjectURL(objectUrl); console.log("Revoked previous object URL"); objectUrl = null; }
        objectUrl = URL.createObjectURL(file); // Create new URL

        if (file.type.startsWith('image/')) {
            previewImage = new Image();
            previewImage.onload = () => { initializeCanvas(previewImage.naturalWidth, previewImage.naturalHeight); redrawCanvas(true); };
            previewImage.onerror = previewLoadError;
            previewImage.src = objectUrl;
        } else if (file.type.startsWith('video/')) {
            previewImage = document.createElement('video');
            previewImage.onloadedmetadata = () => { initializeCanvas(previewImage.videoWidth, previewImage.videoHeight); previewImage.currentTime = 0; };
             previewImage.onseeked = () => { redrawCanvas(true); }; // Draw first frame
             previewImage.onerror = previewLoadError;
             previewImage.muted = true; previewImage.playsInline = true; previewImage.preload = 'metadata';
             previewImage.src = objectUrl;
             previewImage.load();
        } else { previewLoadError(); }
    }
    function previewLoadError(e) { console.error("Error loading preview media:", e); canvasContainer.innerHTML = '<p style="color: red;">Could not load preview for drawing.</p>'; previewPlaceholder.style.display = 'none'; canvas = null; ctx = null; previewImage = null; drawnRegions = []; if(objectUrl) URL.revokeObjectURL(objectUrl); objectUrl=null; }
    function initializeCanvas(originalWidth, originalHeight) { /* (Unchanged) */ if (canvas) { canvas.remove(); } canvas = document.createElement('canvas'); const containerWidth = canvasContainer.clientWidth > 0 ? canvasContainer.clientWidth - 20 : 500; const containerHeight = 450; let displayWidth = originalWidth; let displayHeight = originalHeight; scaleFactor = 1; if (originalWidth > containerWidth) { scaleFactor = containerWidth / originalWidth; displayWidth = containerWidth; displayHeight = originalHeight * scaleFactor; } if (displayHeight > containerHeight) { scaleFactor = containerHeight / originalHeight; displayWidth = originalWidth * scaleFactor; displayHeight = containerHeight; } canvas.width = displayWidth; canvas.height = displayHeight; canvasContainer.appendChild(canvas); ctx = canvas.getContext('2d'); console.log(`Canvas initialized. Original: ${originalWidth}x${originalHeight}, Display: ${displayWidth.toFixed(0)}x${displayHeight.toFixed(0)}, Scale: ${scaleFactor.toFixed(3)}`); canvas.addEventListener('mousedown', startDraw); canvas.addEventListener('mousemove', draw); canvas.addEventListener('mouseup', stopDraw); canvas.addEventListener('mouseleave', stopDraw); canvas.addEventListener('touchstart', startDraw, { passive: false }); canvas.addEventListener('touchmove', draw, { passive: false }); canvas.addEventListener('touchend', stopDraw); canvas.addEventListener('touchcancel', stopDraw); }
    function redrawCanvas(drawImage = false) { /* (Unchanged) */ if (!canvas || !ctx) return; ctx.clearRect(0, 0, canvas.width, canvas.height); if (drawImage && previewImage) { try { if (previewImage instanceof HTMLImageElement && previewImage.complete) { ctx.drawImage(previewImage, 0, 0, canvas.width, canvas.height); } else if (previewImage instanceof HTMLVideoElement && previewImage.readyState >= 1) { ctx.drawImage(previewImage, 0, 0, canvas.width, canvas.height); } } catch (e) { console.error("Error drawing base image/frame:", e); } } else if (!drawImage && previewImage) { try { ctx.drawImage(previewImage, 0, 0, canvas.width, canvas.height); } catch(e){} } ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)'; ctx.lineWidth = 2; ctx.fillStyle = 'rgba(255, 0, 0, 0.2)'; drawnRegions.forEach(rect => { ctx.fillRect(rect.x * scaleFactor, rect.y * scaleFactor, rect.w * scaleFactor, rect.h * scaleFactor); ctx.strokeRect(rect.x * scaleFactor, rect.y * scaleFactor, rect.w * scaleFactor, rect.h * scaleFactor); }); if (isDrawing && currentRect) { ctx.fillStyle = 'rgba(255, 0, 0, 0.3)'; ctx.fillRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h); ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h); } updateRegionCountDisplay(); }
    let currentRect = null;
    function getCanvasPos(evt) { /* (Unchanged) */ if(!canvas) return null; const rect = canvas.getBoundingClientRect(); const clientX = evt.clientX ?? evt.touches?.[0]?.clientX; const clientY = evt.clientY ?? evt.touches?.[0]?.clientY; if (clientX === undefined || clientY === undefined) return null; return { x: (clientX - rect.left) * (canvas.width / rect.width), y: (clientY - rect.top) * (canvas.height / rect.height) }; }
    function startDraw(e) { /* (Unchanged) */ if (!canvas || !ctx) return; isDrawing = true; const pos = getCanvasPos(e); if (!pos) { isDrawing = false; return; } startX = pos.x; startY = pos.y; currentRect = { x: startX, y: startY, w: 0, h: 0 }; e.preventDefault(); }
    function draw(e) { /* (Unchanged) */ if (!isDrawing || !canvas || !ctx) return; const pos = getCanvasPos(e); if (!pos) return; const currentX = pos.x; const currentY = pos.y; currentRect.x = Math.min(startX, currentX); currentRect.y = Math.min(startY, currentY); currentRect.w = Math.abs(startX - currentX); currentRect.h = Math.abs(startY - currentY); redrawCanvas(false); e.preventDefault(); }
    function stopDraw(e) { /* (Unchanged) */ if (!isDrawing || !currentRect) return; isDrawing = false; if (currentRect.w > 5 && currentRect.h > 5) { const originalX = Math.round(currentRect.x / scaleFactor); const originalY = Math.round(currentRect.y / scaleFactor); const originalW = Math.round(currentRect.w / scaleFactor); const originalH = Math.round(currentRect.h / scaleFactor); drawnRegions.push({ x: originalX, y: originalY, w: originalW, h: originalH }); console.log("Region Added (Original Coords):", drawnRegions[drawnRegions.length-1]); } else { console.log("Region too small, discarded."); } currentRect = null; redrawCanvas(true); }
    clearDrawingsBtn.addEventListener('click', clearCanvasAndRegions);
    function clearCanvasAndRegions() { /* (Unchanged) */ drawnRegions = []; currentRect = null; isDrawing = false; redrawCanvas(true); }
    function updateRegionCountDisplay() { regionCountDisplay.textContent = `Regions drawn: ${drawnRegions.length}`; }

    // --- Reset Function ---
     function resetAllStates() {
        // (Now called by Upload New button)
        currentMediaFile = null; fileNameDisplay.textContent = 'No file selected'; fileInput.value = '';
        clearCanvasAndRegions();
        if(canvas) canvas.remove(); canvas = null; ctx = null; previewImage = null;
        if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; console.log("Revoked object URL on reset."); }
        previewPlaceholder.style.display = 'block'; drawPreviewSection.style.display = 'none';
        outputArea.innerHTML = '<p>Processed output will appear here.</p>';
        resultActionsDiv.style.display = 'none'; // Hide result buttons
        processingIndicator.style.display = 'none'; processBtn.disabled = true;
        setControlsState(false); // Re-enable controls
        if (status_label) status_label.textContent = "Ready. Load media.";
     }

     // --- Add Event Listener for Upload New Button ---
     uploadNewBtn.addEventListener('click', resetAllStates);


     // --- Processing ---
    processBtn.addEventListener('click', () => {
        // (Includes status_label checks)
        if (!currentMediaFile) { alert('Please select an image or video file first.'); return; }
        processingIndicator.style.display = 'block'; outputArea.innerHTML = ''; resultActionsDiv.style.display = 'none'; // Hide result buttons during processing
        processBtn.disabled = true; setControlsState(true);
        if (status_label) status_label.textContent = "Processing started...";
        const formData = new FormData(); formData.append('file', currentMediaFile); formData.append('blurFactor', blurSlider.value);
        const regionsToSend = drawnRegions.map(r => [r.x, r.y, r.w, r.h]);
        formData.append('manualRegions', JSON.stringify(regionsToSend));
        console.log("Sending manual regions:", JSON.stringify(regionsToSend));
        fetch('/process', { method: 'POST', body: formData })
        .then(response => {
            if (!response.ok) { return response.json().then(errData => { throw new Error(errData.error || `Server error: ${response.statusText}`); }).catch(() => { throw new Error(`Server error: ${response.statusText} (${response.status})`); }); }
            return response.json();
        })
        .then(data => {
            processingIndicator.style.display = 'none';
            setControlsState(false); // Re-enable general controls now
            if (data.success) {
                displayResult(data.url, data.is_video); downloadLink.href = data.url; downloadLink.download = `blurred_${currentMediaFile.name}`;
                resultActionsDiv.style.display = 'flex'; // Show result buttons (Download + Upload New)
                const methodInfo = data.method_used ? ` (${data.method_used})` : '';
                const successMsg = `Processing complete${methodInfo}. ${data.regions_processed} region(s) blurred in ${data.processing_time}s.`;
                if (status_label) status_label.textContent = successMsg;
                console.log(successMsg);
                // Keep process button disabled until a *new* file is loaded
                 processBtn.disabled = true;
            } else {
                const errorMsg = `Error: ${data.error}`;
                outputArea.innerHTML = `<p style="color: red;">${errorMsg}</p>`;
                if (status_label) status_label.textContent = `Processing failed: ${data.error}`;
                processBtn.disabled = true; // Keep disabled on error
            }
        })
        .catch(error => {
            console.error('Error processing file:', error);
            processingIndicator.style.display = 'none';
            const errorMsg = `An error occurred: ${error.message}. Check console.`;
            outputArea.innerHTML = `<p style="color: red;">${errorMsg}</p>`;
            if (status_label) status_label.textContent = `Error: ${error.message}`;
            setControlsState(false); // Re-enable controls
            processBtn.disabled = true; // Keep disabled on error
        });
    });

    // Helper to enable/disable specific controls during processing
     function setControlsState(disabled) {
        // Disables/Enables controls that should not be used during processing
        const fileInputElement = document.getElementById('fileElem');
        // const processButtonElement = document.getElementById('process-btn'); // Controlled separately
        const dropAreaElement = document.getElementById('drop-area');
        const blurSliderElement = document.getElementById('blur-slider');
        const clearDrawingsButtonElement = document.getElementById('clear-drawings-btn');
        const browseButtonSpan = document.querySelector('#upload-section .browse-btn'); // Get browse span

        if(fileInputElement) fileInputElement.disabled = disabled;
        // if(processButtonElement) processButtonElement.disabled = disabled;
        if(dropAreaElement) { dropAreaElement.style.opacity = disabled ? '0.6' : '1'; dropAreaElement.style.cursor = disabled ? 'not-allowed' : 'pointer'; dropAreaElement.style.pointerEvents = disabled ? 'none' : 'auto'; }
        if(blurSliderElement) blurSliderElement.disabled = disabled;
        if(clearDrawingsButtonElement) clearDrawingsButtonElement.disabled = disabled;
        // Make browse span look disabled too
        if(browseButtonSpan) browseButtonSpan.style.opacity = disabled ? '0.6' : '1';
        if(browseButtonSpan) browseButtonSpan.style.cursor = disabled ? 'not-allowed' : 'pointer';
     }

    function displayResult(url, isVideo) {
        // Displays processed image or video in the output area
        outputArea.innerHTML = ''; // Clear placeholder
        if (isVideo) {
            const video = document.createElement('video');
            video.controls = true; // Add default browser controls
            video.style.maxWidth = '100%'; video.style.maxHeight = '500px'; // Limit size
            video.preload = 'metadata'; // Hint to load duration etc.
            const source = document.createElement('source'); source.src = url;
            // Guess type based on URL if needed, otherwise let browser figure it out
            const extension = url.split('.').pop().toLowerCase();
            if (['mp4', 'webm', 'ogg'].includes(extension)) {
                 source.type = `video/${extension}`;
            } else {
                 source.type = 'video/mp4'; // Default guess for broad compatibility
            }
            video.appendChild(source); outputArea.appendChild(video);
        } else {
            const img = document.createElement('img');
            img.src = url; img.alt = 'Processed Image';
            outputArea.appendChild(img);
        }
    }

}); // End DOMContentLoaded