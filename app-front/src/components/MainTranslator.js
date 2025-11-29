import React, { useState, useRef } from 'react';

function MainTranslator() {
  const [inputType, setInputType] = useState('text');
  const [inputValue, setInputValue] = useState('');
  const [resultVideoUrl, setResultVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const videoRef = useRef(null);

  const API_BASE = process.env.REACT_APP_API_BASE || '';

  // Handler básico
  const handleInputChange = (e) => setInputValue(e.target.value);

  async function generateVideo() {
    setError(null);
    setLoading(true);
    setResultVideoUrl(null);
    try {
      if (inputType === 'text') {
        const resp = await fetch(`${API_BASE}/generate_from_text/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: inputValue }),
        });
        const j = await resp.json();
        if (!resp.ok) throw new Error(j.detail || 'Server error');
        setResultVideoUrl(API_BASE + j.download_url);
      } else if (inputType === 'link') {
        const resp = await fetch(`${API_BASE}/transcribe_youtube/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: inputValue }),
        });
        const j = await resp.json();
        if (!resp.ok) throw new Error(j.detail || 'Server error');
        setResultVideoUrl(API_BASE + j.download_url);
      } else if (inputType === 'file') {
        if (!inputValue) throw new Error('No file selected');
        const form = new FormData();
        form.append('file', inputValue);
        const resp = await fetch(`${API_BASE}/procesar_video/`, {
          method: 'POST',
          body: form,
        });
        const j = await resp.json();
        if (!resp.ok) throw new Error(j.detail || 'Server error');
        setResultVideoUrl(API_BASE + j.download_url);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function onChangePlaybackRate(e) {
    const r = parseFloat(e.target.value) || 1.0;
    setPlaybackRate(r);
    if (videoRef.current) videoRef.current.playbackRate = r;
  }

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
            <button className="translate-btn" onClick={generateVideo} disabled={loading}>
              {loading ? 'Generando...' : 'Generar vídeo'}
            </button>
            {error && <div className="error">{error}</div>}
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
              <div>
                <video ref={videoRef} src={resultVideoUrl} controls width="100%" />
                <div style={{ marginTop: 8 }}>
                  <label>Velocidad: </label>
                  <select value={playbackRate} onChange={onChangePlaybackRate}>
                    <option value={0.5}>0.5x</option>
                    <option value={0.75}>0.75x</option>
                    <option value={1}>1x</option>
                    <option value={1.25}>1.25x</option>
                    <option value={1.5}>1.5x</option>
                    <option value={2}>2x</option>
                  </select>
                </div>
              </div>
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