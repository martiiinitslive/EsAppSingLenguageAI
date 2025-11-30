import React, { useState, useRef } from 'react';

function MainTranslator() {
  const [inputType, setInputType] = useState('text');
  const [inputValue, setInputValue] = useState('');
  const [resultVideoUrl, setResultVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const videoRef = useRef(null);

  // Default backend URL for development
  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';
  const DEFAULT_YT = 'https://www.youtube.com/watch?v=jNQXAC9IVRw';

  // Basic handler
  const handleInputChange = (e) => setInputValue(e.target.value);

  async function generateVideo() {
    setError(null);
    setLoading(true);
    setResultVideoUrl(null);
    try {
      // Helper to parse response safely and provide meaningful errors
      const parseResponse = async (resp) => {
        const text = await resp.text();
        const ct = resp.headers.get('content-type') || '';
        if (!resp.ok) {
          // try to parse JSON error if present
          if (ct.includes('application/json')) {
            try { const j = JSON.parse(text); throw new Error(j.detail || j.error || JSON.stringify(j)); } catch(e) { throw new Error(text || `HTTP ${resp.status}`); }
          }
          throw new Error(text || `HTTP ${resp.status}`);
        }
        if (ct.includes('application/json')) {
          try { return JSON.parse(text); } catch (e) { throw new Error('Invalid JSON response from server'); }
        }
        // Non-JSON success response: return raw text
        return { raw: text };
      };

      if (inputType === 'text') {
        const resp = await fetch(`${API_BASE}/generate_from_text/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: inputValue }),
        });
        const j = await parseResponse(resp);
        if (j.download_url) setResultVideoUrl(API_BASE + j.download_url);
        else throw new Error('Unexpected server response');
      } else if (inputType === 'link') {
        const resp = await fetch(`${API_BASE}/transcribe_youtube/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: inputValue }),
        });
        const j = await parseResponse(resp);
        if (j.download_url) setResultVideoUrl(API_BASE + j.download_url);
        else throw new Error('Unexpected server response');
      } else if (inputType === 'file') {
        if (!inputValue) throw new Error('No file selected');
        const form = new FormData();
        form.append('file', inputValue);
        const resp = await fetch(`${API_BASE}/procesar_video/`, {
          method: 'POST',
          body: form,
        });
        const j = await parseResponse(resp);
        if (j.download_url) setResultVideoUrl(API_BASE + j.download_url);
        else throw new Error('Unexpected server response');
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
          onClick={() => { setInputType('text'); setInputValue(''); }}
        >
          <span role="img" aria-label="Text">📝</span> Text
        </button>
        <button
          className={inputType === 'file' ? 'active' : ''}
          onClick={() => { setInputType('file'); setInputValue(null); }}
        >
          <span role="img" aria-label="File">🎵</span> Audio/Video
        </button>
        <button
          className={inputType === 'link' ? 'active' : ''}
          onClick={() => { setInputType('link'); setInputValue(DEFAULT_YT); }}
        >
          <span role="img" aria-label="Link">🔗</span> Link
        </button>
      </div>

      {/* Paneles traductor */}
      <div className="translator-container">
        <div className="translator-panel input-panel">
          <div className="panel-header">
            {inputType === 'text' && (
              <>
                <span role="img" aria-label="input">📝</span> Text
              </>
            )}
            {inputType === 'file' && (
              <>
                <span role="img" aria-label="input">🎵</span> Audio/Video
              </>
            )}
            {inputType === 'link' && (
              <>
                <span role="img" aria-label="input">🔗</span> Link
              </>
            )}
          </div>
          <div className="input-area">
            {inputType === 'text' && (
              <textarea
                rows={6}
                placeholder="Write text here..."
                value={inputValue || ''}
                onChange={handleInputChange}
              />
            )}
            {inputType === 'file' && (
              <div className="file-upload-container">
                <label className="file-upload-label">
                  Select file
                  <input
                    type="file"
                    accept="audio/*,video/*"
                    className="file-upload-input"
                    onChange={e => setInputValue(e.target.files[0])}
                  />
                </label>
                <span className="file-upload-name">
                  {inputValue ? inputValue.name : "No file selected"}
                </span>
              </div>
            )}
            {inputType === 'link' && (
              <input
                type="text"
                placeholder="Paste the link here..."
                value={inputValue || DEFAULT_YT}
                onChange={handleInputChange}
              />
            )}
          </div>
            <button className="translate-btn" onClick={generateVideo} disabled={loading}>
              {loading ? 'Generating...' : 'Generate video'}
            </button>
            {error && <div className="error">{error}</div>}
        </div>
        <div className="panel-separator">
          <span role="img" aria-label="arrow">➡️</span>
        </div>
        <div className="translator-panel output-panel">
          <div className="panel-header">
            <span role="img" aria-label="output">🎬</span> Result
          </div>
          <div className="result-area">
            {resultVideoUrl ? (
              <div>
                <video ref={videoRef} src={resultVideoUrl} controls width="100%" />
                <div style={{ marginTop: 8 }}>
                  <label>Speed: </label>
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
                <p>The generated video will appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default MainTranslator;