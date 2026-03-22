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

    // File input listeners for showing selected file name and live preview!
    videoInput.addEventListener('change', (e) => {
        if(e.target.files[0]) {
            videoMsg.textContent = e.target.files[0].name;
            const videoPreview = document.getElementById('video-preview');
            videoPreview.src = URL.createObjectURL(e.target.files[0]);
            videoPreview.classList.remove('hidden');
        }
    });

    logoInput.addEventListener('change', (e) => {
        if(e.target.files[0]) {
            logoMsg.textContent = e.target.files[0].name;
            const logoPreview = document.getElementById('logo-preview');
            logoPreview.src = URL.createObjectURL(e.target.files[0]);
            logoPreview.classList.remove('hidden');
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Loading state
        processBtn.disabled = true;
        btnText.textContent = "Processing... (This may take a minute)";
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('video', videoInput.files[0]);
        formData.append('logo', logoInput.files[0]);

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Processing failed');
            }

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

            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            alert("Error: " + error.message);
        } finally {
            // Restore UI
            processBtn.disabled = false;
            btnText.textContent = "Process Video";
            loader.classList.add('hidden');
        }
    });
});
