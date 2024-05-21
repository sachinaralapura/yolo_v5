import React from "react";
import "./style.css";
import { Link } from "react-router-dom";
function NavBar() {
  return (
    <div>
      <div className="navbar">
        <Link to="/">Home</Link>
        <Link to="/dashboard">Dashboard</Link>
        <Link to="/graph">Graphs</Link>
        <Link to="/about"> about</Link>
      </div>
    </div>
  );
}

export default NavBar;
