import React, { useState } from "react";
import "./App.css";
import ChatWindow from "./components/ChatWindow";
import logo from "./ps-25-year-logo.svg";

function App() {

  return (
    <div className="App">
      <div className="heading">
        <img src={logo} alt="PS 25 Year Logo" className="header-logo" />
        <div className="title">PartSelect AI Assistant (Beta)</div>
        <div className="contact-info">
          <div className="phone-number">1-888-741-7748</div>
          <div className="hours">Monday to Saturday</div>
          <div className="hours">8am-9pm EST</div>
        </div>
      </div>
        <ChatWindow/>
    </div>
  );
}

export default App;
