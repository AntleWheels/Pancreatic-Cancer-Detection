// Add interactivity to enhance user experience
document.addEventListener("DOMContentLoaded", () => {
    const resultBox = document.querySelector('.result-box');
    if (resultBox) {
        resultBox.style.transition = "transform 0.5s ease-in-out, opacity 0.5s";
        resultBox.style.transform = "scale(1.1)";
        resultBox.style.opacity = "0.8";
        setTimeout(() => {
            resultBox.style.transform = "scale(1)";
            resultBox.style.opacity = "1";
        }, 500);
    }
});
