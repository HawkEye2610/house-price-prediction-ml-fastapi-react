import { useState } from "react";
import "./PredictionForm.css";

const API_URL = import.meta.env.VITE_API_URL;

function PredictionForm() {
  const [formData, setFormData] = useState({
    MedInc: "",
    HouseAge: "",
    AveRooms: "",
    AveBedrms: "",
    Population: "",
    AveOccup: "",
    Latitude: "",
    Longitude: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [selectedFile, setSelectedFile] = useState(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [fileError, setFileError] = useState("");


  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handlePredict = async () => {
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          MedInc: Number(formData.MedInc),
          HouseAge: Number(formData.HouseAge),
          AveRooms: Number(formData.AveRooms),
          AveBedrms: Number(formData.AveBedrms),
          Population: Number(formData.Population),
          AveOccup: Number(formData.AveOccup),
          Latitude: Number(formData.Latitude),
          Longitude: Number(formData.Longitude),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        let errorMessage = "Prediction request failed.";

        if (Array.isArray(data.detail)) {
          errorMessage = data.detail
            .map((error) => error.msg)
            .join(", ");
        } else if (typeof data.detail === "string") {
          errorMessage = data.detail;
        }
      
        throw new Error(errorMessage);
      }

      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };


  const handleFileChange = (event) => {
    const file = event.target.files[0];

    setSelectedFile(file || null);
    setFileError("");
  };

  const handleFilePrediction = async () => {
    if (!selectedFile) {
      setFileError("Please select a CSV file first.");
      return;
    }

    setFileLoading(true);
    setFileError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
                             `${API_URL}/predict_file`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        let errorMessage = "File prediction failed.";

        try {
          const errorData = await response.json();
          if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail
              .map((error) => error.msg)
              .join(", ");
          } else if (typeof errorData.detail === "string") {
            errorMessage = errorData.detail;
          }
          
        } catch {
          // Keep the default error message
        }

        throw new Error(errorMessage);
      }

      const blob = await response.blob();

      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      link.href = downloadUrl;
      link.download = "predictions.csv";

      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      setFileError(err.message);
    } finally {
      setFileLoading(false);
    }
  };


  return (
    <div className="prediction-form">
      <div className="form-grid">
        <div className="form-group">
          <label>Median Income</label>
          <input
            type="number"
            name="MedInc"
            value={formData.MedInc}
            onChange={handleChange}
            placeholder="e.g. 5"
          />
        </div>

        <div className="form-group">
          <label>House Age</label>
          <input
            type="number"
            name="HouseAge"
            value={formData.HouseAge}
            onChange={handleChange}
            placeholder="e.g. 20"
          />
        </div>

        <div className="form-group">
          <label>Average Rooms</label>
          <input
            type="number"
            name="AveRooms"
            value={formData.AveRooms}
            onChange={handleChange}
            placeholder="e.g. 6"
          />
        </div>

        <div className="form-group">
          <label>Average Bedrooms</label>
          <input
            type="number"
            name="AveBedrms"
            value={formData.AveBedrms}
            onChange={handleChange}
            placeholder="e.g. 1"
          />
        </div>

        <div className="form-group">
          <label>Population</label>
          <input
            type="number"
            name="Population"
            value={formData.Population}
            onChange={handleChange}
            placeholder="e.g. 1000"
          />
        </div>

        <div className="form-group">
          <label>Average Occupancy</label>
          <input
            type="number"
            name="AveOccup"
            value={formData.AveOccup}
            onChange={handleChange}
            placeholder="e.g. 3"
          />
        </div>

        <div className="form-group">
          <label>Latitude</label>
          <input
            type="number"
            name="Latitude"
            value={formData.Latitude}
            onChange={handleChange}
            placeholder="e.g. 37"
          />
        </div>

        <div className="form-group">
          <label>Longitude</label>
          <input
            type="number"
            name="Longitude"
            value={formData.Longitude}
            onChange={handleChange}
            placeholder="e.g. -122"
          />
        </div>
      </div>

      <button
        className="predict-button"
        type="button"
        onClick={handlePredict}
        disabled={loading}
      >
        {loading ? "Predicting..." : "Predict House Price"}
      </button>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {prediction && (
        <div className="prediction-result">
          <p className="result-label">Estimated House Price</p>
            
          <h3 className="predicted-price">
            ${prediction.predicted_price_usd.toLocaleString()}
          </h3>
            
          <div className="result-details">
            <div className="result-item">
              <span>Estimated Error</span>
              <strong>
                ±${prediction.estimated_error_usd.toLocaleString()}
              </strong>
            </div>
            
            <div className="result-item">
              <span>Estimated Range</span>
              <strong>
                ${prediction.prediction_range.lower.toLocaleString()}
                {" - "}
                ${prediction.prediction_range.upper.toLocaleString()}
              </strong>
            </div>
            
            <div className="result-item">
              <span>Model</span>
              <strong>{prediction.model}</strong>
            </div>
          </div>
        </div>
      )}


      <div className="batch-section">
        <div className="batch-header">
          <div>
            <p className="batch-eyebrow">BATCH PREDICTION</p>
            <h3>Predict multiple houses</h3>
            <p>
              Upload a CSV containing house features and download the
              predictions.
            </p>
          </div>
        </div>

        <div className="file-upload-row">
          <label className="file-input-wrapper">
            <span>
              {selectedFile ? selectedFile.name : "Choose CSV file"}
            </span>

            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
            />
          </label>

          <button
            className="batch-button"
            type="button"
            onClick={handleFilePrediction}
            disabled={fileLoading}
          >
            {fileLoading ? "Processing..." : "Upload & Predict"}
          </button>
        </div>

        {fileError && (
          <div className="error-message">
            {fileError}
          </div>
        )}
      </div>

    </div>
  );
}

export default PredictionForm;