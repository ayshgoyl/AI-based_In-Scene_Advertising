document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const videoInput = document.getElementById('video-upload');
    const logoInput = document.getElementById('logo-upload');
    const videoMsg = document.getElementById('video-msg');
    const logoMsg = document.getElementById('logo-msg');
    const processBtn = document.getElementById('process-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    const resultsSection = document.getElementById('results-section');

    // File input listeners for showing selected file name
    videoInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            videoMsg.textContent = e.target.files[0].name;
            videoMsg.parentElement.parentElement.classList.add('active');
        }
    });

    logoInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            logoMsg.textContent = e.target.files[0].name;
            logoMsg.parentElement.parentElement.classList.add('active');
        }
    });

    const prepareBtn = document.getElementById('prepare-btn');
    const prepareLoader = document.getElementById('prepare-loader');
    const metadataSection = document.getElementById('metadata-section');
    const metadataTbody = document.getElementById('metadata-tbody');
    const processLoader = document.getElementById('process-loader');
    
    let currentJobId = null;

    prepareBtn.addEventListener('click', async () => {
        if (!videoInput.files[0] || !logoInput.files[0]) {
            alert("Please select both a video and a logo.");
            return;
        }

        prepareBtn.disabled = true;
        prepareBtn.textContent = "Uploading & Converting... (This will take a moment)";
        prepareLoader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('video', videoInput.files[0]);
        formData.append('logo', logoInput.files[0]);

        try {
            const response = await fetch('/prepare', { method: 'POST', body: formData });
            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Preparation failed');
            
            currentJobId = data.job_id;
            
            // Bind frontend uploaded assets directly via Object URLs
            document.getElementById('uploaded-video-preview').src = URL.createObjectURL(videoInput.files[0]);
            document.getElementById('uploaded-logo-preview').src = URL.createObjectURL(logoInput.files[0]);
            
            // Build table dynamically
            metadataTbody.innerHTML = `
                <tr><td>Initial Starting Size of Input Video</td><td>${data.table.initial_video_size}</td></tr>
                <tr><td>Initial Starting Codec of Input Video</td><td>${data.table.initial_video_codec}</td></tr>
                <tr><td>Initial Starting FPS of Input Video</td><td>${data.table.initial_video_fps}</td></tr>
                <tr><td>Size of Brand Logo Image</td><td>${data.table.logo_size}</td></tr>
                <tr><td>Dimensions of Brand Logo Image</td><td>${data.table.logo_dims}</td></tr>
                <tr><td colspan="2" style="background:#eaf8ea; font-weight:bold;">↳ FFmpeg Transformation Applied (H.264, 24fps)</td></tr>
                <tr><td>New Transformed Size of Input Video</td><td>${data.table.new_video_size}</td></tr>
                <tr><td>New Transformed Codec of Input Video</td><td>${data.table.new_video_codec}</td></tr>
                <tr><td>New Transformed FPS of Input Video</td><td>${data.table.new_video_fps}</td></tr>
            `;

            metadataSection.classList.remove('hidden');
        } catch(err) {
            alert("Error: " + err.message);
        } finally {
            prepareBtn.disabled = false;
            prepareBtn.textContent = "1. Upload & Prepare Assets (Convert to 24fps H.264)";
            prepareLoader.classList.add('hidden');
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentJobId) {
            alert("Please prepare the assets first.");
            return;
        }

        // UI Loading state
        processBtn.disabled = true;
        processBtn.textContent = "Processing... (This may take a minute)";
        processLoader.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        const fd = new FormData();
        fd.append('job_id', currentJobId);

        try {
            const response = await fetch('/process', { method: 'POST', body: fd });
            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Processing failed');

            // Populate extracted frames recursively!
            const extractedContainer = document.getElementById('res-extracted-container');
            extractedContainer.innerHTML = ''; // reset past jobs sequentially
            if (data.extracted_frames && data.extracted_frames.length > 0) {
                data.extracted_frames.forEach(frameUrl => {
                    const img = document.createElement('img');
                    img.src = frameUrl + "?t=" + new Date().getTime();
                    img.style.width = '120px'; // thumbnail sizing!
                    img.style.height = 'auto';
                    img.style.objectFit = 'contain';
                    img.alt = 'Extracted video frame output';
                    extractedContainer.appendChild(img);
                });
            }

            // Populate other results
            document.getElementById('res-surface').src = data.detected_surface + "?t=" + new Date().getTime();
            document.getElementById('res-warped').src = data.warped_ad + "?t=" + new Date().getTime();
            
            const videoEl = document.getElementById('res-video');
            videoEl.src = data.output_video + "?t=" + new Date().getTime();
            videoEl.load();

            // Show results section
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            alert("Error: " + error.message);
        } finally {
            // Restore UI
            processBtn.disabled = false;
            processBtn.textContent = "2. Run AI Ad Placement";
            processLoader.classList.add('hidden');
        }
    });

    // --- Metrics Evaluation Logic ---
    const calcMetricsBtn = document.getElementById('calc-metrics-btn');
    const gtUploadSection = document.getElementById('gt-upload-section');
    const importGtBtn = document.getElementById('import-gt-btn');
    const gtUploadInput = document.getElementById('gt-upload');
    const evaluateLoader = document.getElementById('evaluate-loader');
    const metricsResults = document.getElementById('metrics-results');
    const metricsTbody = document.getElementById('metrics-tbody');

    if (calcMetricsBtn) {
        calcMetricsBtn.addEventListener('click', () => {
            gtUploadSection.classList.remove('hidden');
            calcMetricsBtn.style.display = 'none';
        });
    }

    if (importGtBtn) {
        importGtBtn.addEventListener('click', () => {
            gtUploadInput.click();
        });
    }

    if (gtUploadInput) {
        gtUploadInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            importGtBtn.disabled = true;
            evaluateLoader.classList.remove('hidden');
            metricsResults.classList.add('hidden');

            const fd = new FormData();
            fd.append('job_id', currentJobId);
            fd.append('ground_truth', file);

            try {
                const response = await fetch('/evaluate', { method: 'POST', body: fd });
                const data = await response.json();

                if (!response.ok) throw new Error(data.error || 'Evaluation failed');

                metricsTbody.innerHTML = `
                    <tr><td>Mean Intersection Over Union (mIOU)</td><td><strong>${data.mIOU}</strong></td></tr>
                    <tr><td>Pixel Classification Accuracy</td><td><strong>${data.PixelAccuracy}</strong></td></tr>
                    <tr><td>Mean Average Precision (mAP@0.5)</td><td><strong>${data.mAP_50}</strong></td></tr>
                    <tr><td colspan="2" style="background:#eaf8ea; font-size: 0.9em; text-align:center;">Successfully evaluated ${data.Frames_Evaluated} frames. Logged into Performance logs.</td></tr>
                `;
                metricsResults.classList.remove('hidden');
            } catch (err) {
                alert("Error calculating metrics: " + err.message);
            } finally {
                importGtBtn.disabled = false;
                evaluateLoader.classList.add('hidden');
            }
        });
    }

});
