document.addEventListener('DOMContentLoaded', () => {
    // --- Common Elements ---
    const spinner = document.getElementById('spinner');
    const resultsContent = document.getElementById('resultsContent');

    // --- Image Upload Mode Elements ---
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');

    // --- Live Camera Mode Elements ---
    const startCamBtn = document.getElementById('startCamBtn');
    const stopCamBtn = document.getElementById('stopCamBtn');
    const videoFeed = document.getElementById('videoFeed');
    const videoPlaceholder = document.getElementById('videoPlaceholder');

    // --- Image Upload Logic ---
    imageUpload.addEventListener('change', () => {
        const file = imageUpload.files[0];
        if (file) {
            // Display preview
            const reader = new FileReader();
            reader.onload = (e) => { imagePreview.src = e.target.result; };
            reader.readAsDataURL(file);
            
            // Automatically start inspection
            handleImageInspection(file);
        }
    });

    async function handleImageInspection(file) {
        // Prepare UI for loading
        resultsContent.innerHTML = '';
        spinner.classList.remove('d-none');
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/inspect_image', { method: 'POST', body: formData });
            const result = await response.json();

            if (response.ok) {
                displayUploadResults(result);
            } else {
                throw new Error(result.error || 'Failed to get a valid response.');
            }
        } catch (error) {
            resultsContent.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        } finally {
            spinner.classList.add('d-none');
        }
    }

    function displayUploadResults(result) {
        const { prediction, confidence } = result;
        let statusClass = '';
        let statusIcon = '';

        if (prediction === 'Correct Assembly') {
            statusClass = 'alert-success';
            statusIcon = '✅';
        } else if (prediction === 'Missing Component') {
            statusClass = 'alert-danger';
            statusIcon = '❌';
        } else if (prediction === 'Additional Component') {
            statusClass = 'alert-warning';
            statusIcon = '⚠️';
        }

        resultsContent.innerHTML = `
            <div class="alert ${statusClass}">
                <h4 class="alert-heading">${statusIcon} Status: ${prediction}</h4>
            </div>
        `;
    }

    // --- Live Camera Logic ---
    startCamBtn.addEventListener('click', () => {
        videoFeed.src = '/video_feed'; // Start the stream
        videoPlaceholder.classList.add('d-none');
        startCamBtn.classList.add('d-none');
        stopCamBtn.classList.remove('d-none');
    });

    stopCamBtn.addEventListener('click', () => {
        videoFeed.src = ''; // Stop the stream
        videoPlaceholder.classList.remove('d-none');
        startCamBtn.classList.remove('d-none');
        stopCamBtn.classList.add('d-none');
    });
});