document
  .getElementById("upload-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById("file-input");
    formData.append("file", fileInput.files[0]);

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        const resultDiv = document.getElementById("prediction-result");
        const predictedClass = data.predicted_class;

        // Check the value of predicted_class and set the background color accordingly
        if (predictedClass === "Negative") {
          resultDiv.style.backgroundColor = "green";
        } else if (predictedClass === "Positive") {
          resultDiv.style.backgroundColor = "red";
        } else {
          // Handle other cases or provide a default color
          resultDiv.style.backgroundColor = "white";
        }
        resultDiv.innerHTML = `Predicted Class: ${predictedClass}<br>Confidence: ${data.confidence}`;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  });

const fileInput = document.getElementById("file-input");
const selectedFileName = document.getElementById("selected-file-name");

fileInput.addEventListener("change", function () {
  selectedFileName.textContent = `Selected File: ${fileInput.files[0].name}`;
  selectedFileName.style.display = "block"; // Show the selected file name
});
