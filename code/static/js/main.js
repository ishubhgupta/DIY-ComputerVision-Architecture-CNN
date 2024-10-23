// Function to convert the uploaded image to grayscale
function convertGrayscale(filename) {
    const grayscaleImage = document.getElementById("grayscaleImage");
    grayscaleImage.src = `/grayscale/${filename}`;
    grayscaleImage.style.display = "block";  // Display the grayscale image
}

// Function to apply blur effect to the uploaded image
function applyBlur(filename) {
    const blurValue = document.getElementById("blurRange").value;
    const blurredImage = document.getElementById("blurredImage");
    blurredImage.src = `/blur/${filename}/${blurValue}`;
    blurredImage.style.display = "block";  // Display the blurred image
}

// Function to apply edge detection to the uploaded image
function applyEdgeDetection(filename) {
    const edgeDetectedImage = document.getElementById("edgeDetectedImage");
    edgeDetectedImage.src = `/edge_detection/${filename}`;
    edgeDetectedImage.style.display = "block";  // Display the edge detected image
}

// Optional: Function to handle image path submission
document.getElementById("imagePathForm").addEventListener("submit", function(event) {
    const imagePath = document.getElementById("imagePath").value;
    
    // Optionally validate the input path
    if (!imagePath) {
        event.preventDefault();  // Prevent form submission if path is empty
        alert("Please enter a valid image path.");
    }
});
