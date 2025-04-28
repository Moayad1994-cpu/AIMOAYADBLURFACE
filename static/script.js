// script.js (Combined Manual Draw + Haar Fallback - v9 Video Preview Fix)
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
    const resultActionsDiv = document.getElementById('result-actions');
    const downloadLink = document.getElementById('download-link');
    const uploadNewBtn = document.getElementById('upload-new-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const body = document.body;
    const status_label = document.getElementById('status-label');

    // --- State Variables ---
    let currentMediaFile = null;
    let canvas = null;
    let ctx = null;
    let previewMediaElement = null; // Renamed: Can be Image or Video element
    let isDrawing = false;
    let startX, startY;
    let drawnRegions = []; // Stores {x, y, w, h} relative to *original* media dimensions
    let scaleFactor = 1;
    let objectUrl = null; // For media preview URL management
    let currentRect = null; // For tracking drawing rectangle

    // --- Initial Setup ---
    if (!status_label) console.error("Status label element not found!");

    // --- Theme Toggle ---
    function applyTheme(theme) { body.classList.remove('light-mode', 'dark-mode'); body.classList.add(theme); themeToggleBtn.innerHTML = (theme === 'dark-mode') ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>'; localStorage.setItem('theme', theme); }
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
        // Added more video types here to match HTML accept attribute and backend ALLOWED_EXTENSIONS_VID
        const allowedImageTypes = ['image/jpeg', 'image/png', 'image/webp'];
        const allowedVideoTypes = ['video/mp4', 'video/webm', 'video/ogg', 'video/avi', 'video/mov', 'video/mkv', 'video/wmv', 'video/flv'];
        const isAllowed = allowedImageTypes.includes(file.type) || allowedVideoTypes.includes(file.type) || file.type.startsWith('image/') || file.type.startsWith('video/'); // Fallback check

        if (!isAllowed) {
            alert(`Unsupported file type: ${file.type || 'unknown'}. Please upload common image (JPG, PNG, WebP) or video (MP4, WebM, Ogg, AVI, MOV etc.) formats.`);
            resetAllStates(); return;
        }
        // Simple Max Size Check (using value from backend default)
        const maxSizeMB = 100; const maxSize = maxSizeMB * 1024 * 1024;
        if (file.size > maxSize) {
            alert(`File is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max size: ${maxSizeMB} MB. (Note: Server limit might differ)`);
            resetAllStates(); return;
        }

        resetAllStates(); // Reset everything before loading new file preview

        currentMediaFile = file;
        fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        setupCanvasForPreview(file); // Setup canvas for drawing preview
        outputArea.innerHTML = '<p>Processed output will appear here.</p>';
        resultActionsDiv.style.display = 'none';
        processingIndicator.style.display = 'none';
        processBtn.disabled = false; // Enable process button
        updateStatus("File ready. Draw regions or click Process.");
    }

    // --- Canvas Setup and Drawing ---
    function setupCanvasForPreview(file) {
        clearCanvasAndRegions(); // Clear previous drawings and revoke old Object URL
        canvasContainer.innerHTML = ''; // Clear previous canvas or error messages
        canvasContainer.appendChild(previewPlaceholder); // Add placeholder back initially
        previewPlaceholder.style.display = 'block';
        drawPreviewSection.style.display = 'block';

        // Revoke previous URL if it exists
        if (objectUrl) {
             URL.revokeObjectURL(objectUrl);
             console.log("Revoked previous object URL");
             objectUrl = null;
        }
        objectUrl = URL.createObjectURL(file); // Create new URL

        if (file.type.startsWith('image/')) {
            previewMediaElement = new Image();
            previewMediaElement.onload = () => {
                console.log("Image loaded successfully.");
                previewPlaceholder.style.display = 'none';
                initializeCanvas(previewMediaElement.naturalWidth, previewMediaElement.naturalHeight);
                redrawCanvas(true); // Draw the image immediately
            };
            previewMediaElement.onerror = (e) => previewLoadError("Could not load image.", e);
            previewMediaElement.src = objectUrl;

        } else if (file.type.startsWith('video/')) {
            previewMediaElement = document.createElement('video');

            // Event listener for metadata loaded (gets dimensions)
            previewMediaElement.onloadedmetadata = () => {
                console.log("Video metadata loaded. Dimensions:", previewMediaElement.videoWidth, previewMediaElement.videoHeight, "ReadyState:", previewMediaElement.readyState);
                previewPlaceholder.style.display = 'none';
                initializeCanvas(previewMediaElement.videoWidth, previewMediaElement.videoHeight);
                // Don't draw here yet, wait for 'canplay' or 'seeked'
            };

            // Event listener for when the first frame is ready to be PLAYED (safer for drawing)
            previewMediaElement.addEventListener('canplay', () => {
                 console.log("Video 'canplay' event fired. ReadyState:", previewMediaElement.readyState);
                 // Ensure canvas is initialized before drawing
                 if (canvas && ctx && previewMediaElement.videoWidth > 0) {
                    // Seek to beginning just to be sure we draw the first frame
                    previewMediaElement.currentTime = 0;
                 } else if (!canvas && previewMediaElement.videoWidth > 0) {
                    // Rare case: Metadata didn't load first? Initialize canvas now.
                    console.warn("Canvas not ready when 'canplay' fired, initializing now.");
                    initializeCanvas(previewMediaElement.videoWidth, previewMediaElement.videoHeight);
                    previewMediaElement.currentTime = 0;
                 } else {
                    console.warn("Could not draw on 'canplay': canvas or video dimensions unavailable.");
                 }
            }, { once: true }); // Use 'once' to only fire once

            // Event listener for when seeking (e.g., after setting currentTime) is complete
            previewMediaElement.addEventListener('seeked', () => {
                console.log("Video 'seeked' event fired. ReadyState:", previewMediaElement.readyState, "CurrentTime:", previewMediaElement.currentTime);
                if (canvas && ctx) {
                   redrawCanvas(true); // Draw the frame now that seeking is done
                } else {
                   console.warn("Canvas not ready when 'seeked' fired.");
                }
            }, { once: false }); // Can fire multiple times if user seeks later, but primary draw is handled


             // Event listener for errors during loading/playback
             previewMediaElement.onerror = (e) => {
                 console.error("Video loading error event:", e);
                 // Determine the specific error if possible
                 let errorDetail = "Unknown video error.";
                 if (previewMediaElement.error) {
                     switch (previewMediaElement.error.code) {
                         case MediaError.MEDIA_ERR_ABORTED: errorDetail = 'Video loading aborted.'; break;
                         case MediaError.MEDIA_ERR_NETWORK: errorDetail = 'Network error during video loading.'; break;
                         case MediaError.MEDIA_ERR_DECODE: errorDetail = 'Video decoding error (unsupported format or corrupted file?).'; break;
                         case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED: errorDetail = 'Video format not supported by this browser.'; break;
                         default: errorDetail = `An unknown error occurred (code ${previewMediaElement.error.code}).`;
                     }
                 }
                 previewLoadError(errorDetail, e);
             };

             // Set attributes and source
             previewMediaElement.muted = true; // Important for autoplay/programmatic control
             previewMediaElement.playsInline = true; // Good practice for mobile
             previewMediaElement.preload = 'metadata'; // Load dimensions first
             previewMediaElement.src = objectUrl;
             previewMediaElement.load(); // Explicitly start loading

        } else {
            // Should not happen due to file type check, but as a fallback
            previewLoadError(`Unsupported file type: ${file.type}`);
        }
    }

    function previewLoadError(errorMessage, event = null) {
        console.error("Error loading preview media:", errorMessage, event);
        // Display error inside the canvas container
        canvasContainer.innerHTML = `<div class="error-message">
                                         Could not load preview for drawing.
                                         <p class="error-detail">${errorMessage}</p>
                                     </div>`;
        previewPlaceholder.style.display = 'none'; // Hide the default placeholder
        canvas = null; ctx = null; previewMediaElement = null; drawnRegions = [];
        if(objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl=null; console.log("Revoked object URL on preview error."); }
        processBtn.disabled = true; // Disable processing if preview failed
        updateStatus(`Preview failed: ${errorMessage}`);
    }

    function initializeCanvas(originalWidth, originalHeight) {
        if (!canvasContainer) return;
        if (canvas) { canvas.remove(); } // Remove existing canvas if any
        canvas = document.createElement('canvas');

        // Calculate display size based on container, maintaining aspect ratio
        const containerStyle = window.getComputedStyle(canvasContainer);
        const containerPadding = parseFloat(containerStyle.paddingLeft) + parseFloat(containerStyle.paddingRight);
        const availableWidth = canvasContainer.clientWidth - containerPadding; // Max width available inside padding
        const maxHeight = 450; // Max height constraint

        scaleFactor = 1;
        let displayWidth = originalWidth;
        let displayHeight = originalHeight;

        if (originalWidth <= 0 || originalHeight <= 0) {
             console.warn("Invalid original dimensions:", originalWidth, originalHeight);
             previewLoadError("Media has invalid dimensions (0x0).");
             return; // Stop if dimensions are invalid
        }

        // Scale down if wider than container
        if (originalWidth > availableWidth) {
            scaleFactor = availableWidth / originalWidth;
            displayWidth = availableWidth;
            displayHeight = originalHeight * scaleFactor;
        }

        // Scale down further if taller than max height
        if (displayHeight > maxHeight) {
            scaleFactor = maxHeight / originalHeight; // Recalculate scale based on height limit
            displayWidth = originalWidth * scaleFactor;
            displayHeight = maxHeight;
        }

        // Ensure dimensions are positive
        displayWidth = Math.max(1, displayWidth);
        displayHeight = Math.max(1, displayHeight);


        canvas.width = displayWidth;
        canvas.height = displayHeight;

        // Clear container and append new canvas
        canvasContainer.innerHTML = ''; // Clear placeholder or old canvas/error
        canvasContainer.appendChild(canvas);
        ctx = canvas.getContext('2d');

        console.log(`Canvas initialized. Original: ${originalWidth}x${originalHeight}, Display: ${displayWidth.toFixed(0)}x${displayHeight.toFixed(0)}, Scale: ${scaleFactor.toFixed(3)}`);

        // Add event listeners for drawing
        canvas.addEventListener('mousedown', startDraw);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDraw);
        canvas.addEventListener('mouseleave', stopDraw);
        // Touch events
        canvas.addEventListener('touchstart', startDraw, { passive: false });
        canvas.addEventListener('touchmove', draw, { passive: false });
        canvas.addEventListener('touchend', stopDraw);
        canvas.addEventListener('touchcancel', stopDraw);
    }

    function redrawCanvas(drawImage = false) {
        if (!canvas || !ctx) { console.warn("redrawCanvas called but canvas or context is missing."); return; }
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw the base image or video frame
        if (drawImage && previewMediaElement) {
            try {
                // Check if it's an image and it's loaded
                if (previewMediaElement instanceof HTMLImageElement && previewMediaElement.complete && previewMediaElement.naturalWidth > 0) {
                    ctx.drawImage(previewMediaElement, 0, 0, canvas.width, canvas.height);
                }
                // Check if it's a video and it has enough data to draw the current frame
                else if (previewMediaElement instanceof HTMLVideoElement && previewMediaElement.readyState >= 2) { // HAVE_CURRENT_DATA
                    ctx.drawImage(previewMediaElement, 0, 0, canvas.width, canvas.height);
                    console.log("Drew video frame to canvas.");
                }
            } catch (e) {
                console.error("Error drawing base image/frame:", e);
                // Optionally display an error on the canvas itself
                 ctx.fillStyle = 'red';
                 ctx.font = '12px sans-serif';
                 ctx.fillText('Error drawing preview', 10, 20);
            }
        } else if (!drawImage && previewMediaElement) {
             // Optimization: If not explicitly told to redraw the base image,
             // but we have one, draw it quickly just to avoid flicker during rectangle draw.
             // This might not be strictly necessary if performance is good.
              try {
                 if (previewMediaElement instanceof HTMLImageElement && previewMediaElement.complete && previewMediaElement.naturalWidth > 0) {
                     ctx.drawImage(previewMediaElement, 0, 0, canvas.width, canvas.height);
                 } else if (previewMediaElement instanceof HTMLVideoElement && previewMediaElement.readyState >= 2) {
                     ctx.drawImage(previewMediaElement, 0, 0, canvas.width, canvas.height);
                 }
              } catch(e){/* ignore draw errors here */}
        }


        // Draw existing regions
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
        drawnRegions.forEach(rect => {
            // Draw rectangle based on original coords scaled to canvas display size
            ctx.fillRect(rect.x * scaleFactor, rect.y * scaleFactor, rect.w * scaleFactor, rect.h * scaleFactor);
            ctx.strokeRect(rect.x * scaleFactor, rect.y * scaleFactor, rect.w * scaleFactor, rect.h * scaleFactor);
        });

        // Draw the rectangle currently being drawn
        if (isDrawing && currentRect) {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)'; // Slightly different fill for current drawing
            ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h); // Draw outline
            ctx.fillRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);   // Draw fill
        }

        updateRegionCountDisplay();
    }

    function getCanvasPos(evt) {
        if(!canvas) return null;
        const rect = canvas.getBoundingClientRect();
        // Handle both mouse and touch events
        const clientX = evt.clientX ?? evt.touches?.[0]?.clientX;
        const clientY = evt.clientY ?? evt.touches?.[0]?.clientY;
        if (clientX === undefined || clientY === undefined) return null; // No coordinates available

        // Calculate position relative to canvas, considering scaling
        const canvasX = (clientX - rect.left) * (canvas.width / rect.width);
        const canvasY = (clientY - rect.top) * (canvas.height / rect.height);
        return { x: canvasX, y: canvasY };
    }

    function startDraw(e) {
        if (!canvas || !ctx || !previewMediaElement) return; // Need media loaded to draw
        isDrawing = true;
        const pos = getCanvasPos(e);
        if (!pos) { isDrawing = false; return; }
        startX = pos.x; startY = pos.y;
        // Initialize rectangle being drawn (coords relative to canvas display)
        currentRect = { x: startX, y: startY, w: 0, h: 0 };
        e.preventDefault(); // Prevent default actions like text selection or page scrolling on touch
    }

    function draw(e) {
        if (!isDrawing || !canvas || !ctx || !currentRect) return;
        const pos = getCanvasPos(e);
        if (!pos) return;
        const currentX = pos.x;
        const currentY = pos.y;
        // Update current rectangle dimensions (relative to canvas display)
        currentRect.x = Math.min(startX, currentX);
        currentRect.y = Math.min(startY, currentY);
        currentRect.w = Math.abs(startX - currentX);
        currentRect.h = Math.abs(startY - currentY);
        // Redraw canvas (optimization: don't redraw base image, just overlay rectangles)
        redrawCanvas(false);
        e.preventDefault();
    }

    function stopDraw(e) {
        if (!isDrawing || !currentRect) return;
        isDrawing = false;
        // Convert final canvas coords to original media coords before storing
        // Only add if the rectangle is reasonably sized
        if (currentRect.w > 5 && currentRect.h > 5) {
             const originalX = Math.round(currentRect.x / scaleFactor);
             const originalY = Math.round(currentRect.y / scaleFactor);
             const originalW = Math.round(currentRect.w / scaleFactor);
             const originalH = Math.round(currentRect.h / scaleFactor);
             // Add validation: Ensure coords are within original media bounds
             // (Optional but good practice, though backend should handle clipping too)
             // E.g., Clamp originalX, originalY, originalW, originalH
             drawnRegions.push({ x: originalX, y: originalY, w: originalW, h: originalH });
             console.log("Region Added (Original Coords):", drawnRegions[drawnRegions.length-1]);
        } else {
            console.log("Region too small, discarded.");
        }
        currentRect = null; // Reset current drawing rectangle
        // Redraw canvas completely including base image and all stored regions
        redrawCanvas(true);
    }

    clearDrawingsBtn.addEventListener('click', clearCanvasAndRegions);

    function clearCanvasAndRegions() {
        drawnRegions = [];
        currentRect = null;
        isDrawing = false;
        // Only redraw if canvas exists
        if (canvas && ctx) {
            redrawCanvas(true);
        }
        updateRegionCountDisplay();
    }

    function updateRegionCountDisplay() {
         regionCountDisplay.textContent = `Regions drawn: ${drawnRegions.length}`;
    }

     // --- Status Bar Update ---
     function updateStatus(message) {
         if (status_label) {
            status_label.textContent = `Status: ${message}`;
         } else {
            console.log(`Status: ${message}`);
         }
     }

    // --- Reset Function ---
     function resetAllStates() {
        currentMediaFile = null; fileNameDisplay.textContent = 'No file selected'; fileInput.value = ''; // Clear file input
        clearCanvasAndRegions(); // Clear drawings array and update display
        if(canvas) canvas.remove(); canvas = null; ctx = null; // Remove canvas element
        // Stop video/audio if playing and remove element reference
        if (previewMediaElement && typeof previewMediaElement.pause === 'function') {
            previewMediaElement.pause();
            previewMediaElement.removeAttribute('src'); // Remove src to free resource
            previewMediaElement.load(); // Abort loading/playback
        }
        previewMediaElement = null; // Clear reference
        // Revoke object URL
        if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; console.log("Revoked object URL on reset."); }

        // Reset UI elements
        canvasContainer.innerHTML = ''; // Clear canvas container
        canvasContainer.appendChild(previewPlaceholder); // Add placeholder back
        previewPlaceholder.style.display = 'block';
        drawPreviewSection.style.display = 'none'; // Hide drawing section
        outputArea.innerHTML = '<p>Processed output will appear here.</p>'; // Reset output
        resultActionsDiv.style.display = 'none'; // Hide result buttons
        processingIndicator.style.display = 'none'; // Hide spinner
        processBtn.disabled = true; // Disable process button
        setControlsState(false); // Re-enable general controls
        updateStatus("Ready. Load media."); // Update status bar
     }

     // --- Event Listener for Upload New Button ---
     uploadNewBtn.addEventListener('click', resetAllStates);


     // --- Processing ---
    processBtn.addEventListener('click', () => {
        if (!currentMediaFile) { alert('Please select an image or video file first.'); return; }

        processingIndicator.style.display = 'block'; // Show spinner
        outputArea.innerHTML = ''; // Clear previous results
        resultActionsDiv.style.display = 'none'; // Hide result buttons during processing
        processBtn.disabled = true; // Disable process button itself
        setControlsState(true); // Disable other controls
        updateStatus("Processing started...");

        const formData = new FormData();
        formData.append('file', currentMediaFile);
        formData.append('blurFactor', blurSlider.value);

        // Convert drawnRegions ({x,y,w,h} objects) to array of arrays for JSON
        const regionsToSend = drawnRegions.map(r => [r.x, r.y, r.w, r.h]);
        formData.append('manualRegions', JSON.stringify(regionsToSend));

        console.log("Sending request to /process...");
        console.log(" -> Blur Factor:", blurSlider.value);
        console.log(" -> Manual Regions:", JSON.stringify(regionsToSend));


        fetch('/process', { method: 'POST', body: formData })
        .then(response => {
            if (!response.ok) {
                 // Try to parse error JSON from server, otherwise use status text
                 return response.json().then(errData => {
                     throw new Error(errData.error || `Server error: ${response.statusText} (${response.status})`);
                 }).catch(() => { // If parsing JSON fails
                     throw new Error(`Server error: ${response.statusText} (${response.status})`);
                 });
            }
            return response.json(); // Parse success JSON
        })
        .then(data => {
            processingIndicator.style.display = 'none'; // Hide spinner
            setControlsState(false); // Re-enable general controls
            if (data.success) {
                displayResult(data.url, data.is_video);
                downloadLink.href = data.url;
                // Create a safer filename for download
                const safeOriginalName = currentMediaFile.name.replace(/[^a-z0-9.]/gi, '_').toLowerCase();
                downloadLink.download = `blurred_${safeOriginalName}`;

                resultActionsDiv.style.display = 'flex'; // Show result buttons
                const methodInfo = data.method_used ? ` (${data.method_used})` : '';
                const successMsg = `Processing complete${methodInfo}. ${data.regions_processed} region(s) blurred in ${data.processing_time}s.`;
                updateStatus(successMsg);
                console.log(successMsg);
                // Keep process button disabled until a *new* file is loaded (user must use Upload New)
                 processBtn.disabled = true;
            } else {
                // Handle specific error from server JSON
                const errorMsg = `Error: ${data.error || 'Unknown processing error.'}`;
                outputArea.innerHTML = `<p style="color: red;">${errorMsg}</p>`;
                updateStatus(`Processing failed: ${data.error || 'Unknown error'}`);
                processBtn.disabled = true; // Keep disabled on error
            }
        })
        .catch(error => {
            // Handle fetch errors or errors thrown from response check
            console.error('Error during fetch /process:', error);
            processingIndicator.style.display = 'none'; // Hide spinner
            const errorMsg = `An error occurred: ${error.message}. Check browser console and server logs.`;
            outputArea.innerHTML = `<p style="color: red;">${errorMsg}</p>`;
            updateStatus(`Error: ${error.message}`);
            setControlsState(false); // Re-enable controls
            processBtn.disabled = true; // Keep disabled on error
        });
    });

    // Helper to enable/disable specific controls during processing
     function setControlsState(disabled) {
        // Disables/Enables controls that should not be used during processing
        if(fileInput) fileInput.disabled = disabled;
        if(dropArea) { dropArea.style.opacity = disabled ? '0.6' : '1'; dropArea.style.cursor = disabled ? 'not-allowed' : 'pointer'; dropArea.style.pointerEvents = disabled ? 'none' : 'auto'; }
        if(blurSlider) blurSlider.disabled = disabled;
        if(clearDrawingsBtn) clearDrawingsBtn.disabled = disabled;
        // Make browse button look disabled too (via label opacity/cursor)
        const fileLabel = dropArea ? dropArea.querySelector('.file-label') : null;
        if(fileLabel) { fileLabel.style.cursor = disabled ? 'not-allowed' : 'default'; fileLabel.style.opacity = disabled ? 0.6 : 1; }
        // Disable drawing on canvas
        if(canvas) canvas.style.pointerEvents = disabled ? 'none' : 'auto';
     }

    function displayResult(url, isVideo) {
        // Displays processed image or video in the output area
        outputArea.innerHTML = ''; // Clear placeholder or previous result
        if (isVideo) {
            const video = document.createElement('video');
            video.controls = true; // Add default browser controls
            video.style.maxWidth = '100%'; video.style.maxHeight = '500px'; // Limit size
            video.preload = 'metadata'; // Hint to load duration etc.
            const source = document.createElement('source'); source.src = url;
            // Try to infer type from extension for better compatibility
            const extension = url.split('.').pop().toLowerCase();
             if (['mp4', 'webm', 'ogg'].includes(extension)) {
                 source.type = `video/${extension}`;
             } else if (extension === 'mov') {
                 source.type = 'video/quicktime';
             } else {
                 // Fallback for others like avi, mkv which might not play natively
                 // Browser might still try based on server mime-type
                 source.type = 'video/mp4'; // Common fallback
                 console.warn(`Result video type '${extension}' might not be natively supported by the browser.`);
             }
            video.appendChild(source);
            // Add a fallback message
            video.appendChild(document.createTextNode('Your browser does not support the video tag or this video format.'));
            outputArea.appendChild(video);
        } else {
            const img = document.createElement('img');
            img.src = url; img.alt = 'Processed Image Result';
            img.onerror = () => { outputArea.innerHTML = '<p style="color: red;">Error loading processed image result.</p>'; };
            outputArea.appendChild(img);
        }
    }

}); // End DOMContentLoaded
