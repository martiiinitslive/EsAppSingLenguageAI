import React from 'react';
import MainTranslator from './components/MainTranslator';
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header>
        <h1>Sign2Text</h1>
        <p>Multimedia translator: text, audio, video or Youtube link</p>
      </header>
      <MainTranslator />
    </div>
  );
}

export default App;
