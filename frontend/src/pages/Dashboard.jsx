import React from "react";
import ImageStream from "../components/Stream";
import "./pageStyle.css";
function Dashboard() {
  return (
    <div className="dashboard">
      <div className="videostream">
        Video stream
        <ImageStream />
      </div>
      <div className="faces">faces</div>
    </div>
  );
}

export default Dashboard;
