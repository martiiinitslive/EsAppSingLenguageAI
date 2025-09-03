import React, { useState } from 'react';

function MainTranslator() {
  const [inputType, setInputType] = useState('text');
  const [inputValue, setInputValue] = useState('');
  const [resultVideoUrl, setResultVideoUrl] = useState(null);

  // Handler básico
  const handleInputChange = (e) => setInputValue(e.target.value);

  return (
    <div>
      {/* Barra de selección de tipo de entrada */}
      <div className="input-type-bar">
        <button
          className={inputType === 'text' ? 'active' : ''}
          onClick={() => setInputType('text')}
        >
          <span role="img" aria-label="Texto">📝</span> Texto
        </button>
        <button
          className={inputType === 'file' ? 'active' : ''}
          onClick={() => setInputType('file')}
        >
          <span role="img" aria-label="Archivo">🎵</span> Audio/Vídeo
        </button>
        <button
          className={inputType === 'link' ? 'active' : ''}
          onClick={() => setInputType('link')}
        >
          <span role="img" aria-label="Enlace">🔗</span> Enlace
        </button>
      </div>

      {/* Paneles traductor */}
      <div className="translator-container">
        <div className="translator-panel input-panel">
          <div className="panel-header">
            {inputType === 'text' && (
              <>
                <span role="img" aria-label="input">📝</span> Texto
              </>
            )}
            {inputType === 'file' && (
              <>
                <span role="img" aria-label="input">🎵</span> Audio/Vídeo
              </>
            )}
            {inputType === 'link' && (
              <>
                <span role="img" aria-label="input">🔗</span> Enlace
              </>
            )}
          </div>
          <div className="input-area">
            {inputType === 'text' && (
              <textarea
                rows={6}
                placeholder="Escribe el texto aquí..."
                value={inputValue}
                onChange={handleInputChange}
              />
            )}
            {inputType === 'file' && (
              <div className="file-upload-container">
                <label className="file-upload-label">
                  Seleccionar archivo
                  <input
                    type="file"
                    accept="audio/*,video/*"
                    className="file-upload-input"
                    onChange={e => setInputValue(e.target.files[0])}
                  />
                </label>
                <span className="file-upload-name">
                  {inputValue ? inputValue.name : "Ningún archivo seleccionado"}
                </span>
              </div>
            )}
            {inputType === 'link' && (
              <input
                type="text"
                placeholder="Pega el enlace aquí..."
                value={inputValue}
                onChange={handleInputChange}
              />
            )}
          </div>
          <button className="translate-btn">Generar vídeo</button>
        </div>
        <div className="panel-separator">
          <span role="img" aria-label="arrow">➡️</span>
        </div>
        <div className="translator-panel output-panel">
          <div className="panel-header">
            <span role="img" aria-label="output">🎬</span> Resultado
          </div>
          <div className="result-area">
            {resultVideoUrl ? (
              <video src={resultVideoUrl} controls width="100%" />
            ) : (
              <div className="placeholder">
                <span role="img" aria-label="waiting">⌛</span>
                <p>El vídeo generado aparecerá aquí</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default MainTranslator;