document.getElementById("upload-form").addEventListener("submit", function() {
    document.getElementById("loading").style.display = "flex";
});

function previewFile() {
    const preview = document.getElementById('preview-image');
    const file = document.getElementById('file-input').files[0];
    const reader = new FileReader();

    reader.onloadend = function() {
        preview.src = reader.result;
    };

    if (file) {
        reader.readAsDataURL(file);
    }
}
function previewFile() {
    const fileInput = document.getElementById("file-input");
    const previewImage = document.getElementById("preview-image");

    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else {
        // Reset to default placeholder if no file is selected
        previewImage.src = "/static/images/placeholder.png";
    }
}
