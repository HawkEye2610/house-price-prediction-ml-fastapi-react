import Header from "./components/Header";
import PredictionForm from "./components/PredictionForm";
import "./App.css";

function App() {
  return (
    <div className="app">
      <Header />

      <main className="main-content">
        <section className="hero">

          <span className="hero-badge">
            AI-POWERED PROPERTY ESTIMATION
          </span>
          
          <h2>
            Predict California House Prices
          </h2>
          
          <p>
            Enter property characteristics and get an estimated
            market value powered by a tuned Random Forest model.
          </p>
          
        </section>

        <section className="prediction-card">
          <PredictionForm />
        </section>
      </main>
    </div>
  );
}

export default App;