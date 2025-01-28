document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submit-btn');
    
    form.addEventListener('submit', (event) => {
        // Show loading spinner and disable button during submission
        loading.style.display = 'flex';
        submitBtn.disabled = true;
        submitBtn.style.cursor = 'not-allowed';
    });
});
