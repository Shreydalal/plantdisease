document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const output = document.getElementById("output");
    const loadingSpinner = document.getElementById("loading-spinner");
    const progressBar = document.getElementById("progress-bar");
    const progressFill = document.getElementById("progress-fill");
    const classResult = document.getElementById("class");
    const confidenceResult = document.getElementById("confidence");
    const imagePreview = document.getElementById("image-preview");
    const imagePreviewContainer = document.getElementById("image-preview-container");
    const errorMessage = document.getElementById("error-message");

    // Listen for file input changes
    fileInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            const imageUrl = URL.createObjectURL(file);
            imagePreview.src = imageUrl;
            imagePreviewContainer.classList.remove("hidden"); // Show image preview
        }
    });

    // Handle form submission
    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            alert("Please select an image file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Show the loading spinner and reset output
            loadingSpinner.classList.remove("hidden");
            progressBar.classList.remove("hidden");
            classResult.textContent = "Loading...";
            confidenceResult.textContent = "-";
            output.classList.add("hidden");
            errorMessage.classList.add("hidden");

            // Simulate loading progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 5;
                progressFill.style.width = `${progress}%`;
                if (progress >= 100) clearInterval(progressInterval);
            }, 100);

            // Send request to FastAPI
            const response = await fetch("https://web-plant-disease.up.railway.app/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            // Hide loading indicators
            loadingSpinner.classList.add("hidden");
            progressBar.classList.add("hidden");

            if (data.error) {
                // Handle error
                errorMessage.classList.remove("hidden");
            } else {
                // Update the output with prediction data
                classResult.textContent = data.class;
                confidenceResult.textContent = data.confidence.toFixed(2);
                output.classList.remove("hidden");
            }
        } catch (error) {
            // Hide the spinner and show error message
            loadingSpinner.classList.add("hidden");
            progressBar.classList.add("hidden");
            errorMessage.classList.remove("hidden");
            console.error(error);
        }
    });
});
