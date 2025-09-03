import React from 'react';
import MainTranslator from './components/MainTranslator';
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header>
        <h1>EsApp Sing Language AI</h1>
        <p>Traductor multimedia: texto, audio, vídeo o enlace</p>
      </header>
      <MainTranslator />
    </div>
  );
}

export default App;
