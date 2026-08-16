function Header() {
  return (
    <header className="header">
      <div className="header-content">

        <div className="brand">

          <div className="brand-icon">
            HP
          </div>

          <div className="brand-text">
            <h1>HousePredict</h1>
            <p>Intelligent property estimation</p>
          </div>

        </div>

        <div className="tech-badge">
          ML • FastAPI • React
        </div>

      </div>
    </header>
  );
}

export default Header;