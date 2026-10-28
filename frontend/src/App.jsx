import { useEffect, useRef, useState } from "react";
import axios from "axios";
import Login from "./Login";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:4000";

export default function App() {
  const [token, setToken] = useState(
  () => sessionStorage.getItem("access_token") || ""
);

function handleLogin(accessToken) {
  sessionStorage.setItem("access_token", accessToken);
  setToken(accessToken);
}

function handleLogout() {
  sessionStorage.removeItem("access_token");
  setToken("");
}
  const [file, setFile] = useState(null), [preview, setPreview] = useState("");
  const [result, setResult] = useState(null), [error, setError] = useState(""), [loading, setLoading] = useState(false);
  const inputRef = useRef(null);
  useEffect(() => {
    if (!file) { setPreview(""); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  function choose(f) {
    setError(""); setResult(null);
    if (!f) return;
    if (!f.type.startsWith("image/")) return setError("Please select an image file.");
    if (f.size > 10*1024*1024) return setError("Image must be smaller than 10 MB.");
    setFile(f);
  }
  async function analyze() {
    if (!file) return setError("Upload a fundus image first.");
    setLoading(true); setError(""); setResult(null);
    const body = new FormData(); body.append("image", file);
    try { const r = await axios.post(`${API_URL}/api/screen`, body, { timeout: 130000 }); setResult(r.data); }
    catch (e) { setError(e.response?.data?.error || "Could not analyze image. Check that backend and ML service are running."); }
    finally { setLoading(false); }
  }

function reset() {
  setFile(null);
  setResult(null);
  setError("");
  if (inputRef.current) inputRef.current.value = "";
}

if (!token) {
  return <Login onLogin={handleLogin} />;
}

const demo = result?.mode !== "model";


 
  return <div className="shell">

<header>
  <a className="brand" href="#">
    <span className="logo">R</span> Retina
    <span className="mint">Lens</span>
  </a>

  <div className="header-right">
    <span className="top-tag">● RESEARCH PROTOTYPE</span>
    <button className="link" onClick={handleLogout}>
      Sign out
    </button>
  </div>
</header>

    <main>
      <section className="hero"><div className="eyebrow">EXPLAINABLE AI · RETINAL SCREENING</div>
        <h1>See beyond the <span>prediction.</span></h1>
        <p>An educational prototype exploring diabetic-retinopathy grading, image-quality checks, and visual explanations from retinal fundus images.</p>
        <div className="safety"><b>ⓘ Research use only.</b> This tool is not a diagnosis and must not guide treatment decisions.</div>
      </section>
      <section className="panel">
        <div className="section-title"><h2><small>01</small> Upload fundus image</h2><span>JPG · PNG · WEBP · Max 10 MB</span></div>
        <div className="drop" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();choose(e.dataTransfer.files?.[0]);}}>
          {preview ? <div className="preview"><img src={preview} alt="Selected image preview"/><div><span className="kicker">IMAGE SELECTED</span><b>{file?.name}</b><span className="muted">{(file.size/1048576).toFixed(2)} MB</span><button className="link" onClick={()=>inputRef.current?.click()}>Choose another image</button></div></div> :
          <div className="empty"><div className="upload-icon">↑</div><h3>Drop your image here</h3><p>Choose a retinal fundus photograph from your device.</p><button className="secondary" onClick={()=>inputRef.current?.click()}>Browse files</button><span className="muted tiny">Use public research images or images you have permission to process.</span></div>}
          <input ref={inputRef} hidden type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={e=>choose(e.target.files?.[0])}/>
        </div>
        {error && <div className="error" role="alert">{error}</div>}
        <div className="actions"><button className="primary" disabled={!file||loading} onClick={analyze}>{loading?"Processing image…":"Analyze image →"}</button><button className="link" onClick={reset} disabled={loading}>Reset</button></div>
        {result && <section className="results" aria-live="polite">
          <div className="section-title"><h2><small>02</small> Screening output</h2><span className={`pill ${demo?"warn":"ok"}`}>{demo?"DEMO MODE":"MODEL INFERENCE"}</span></div>
          {demo && <div className="warning"><b>No trained model is installed.</b><p>The upload pipeline works, but this response does not contain a real DR prediction or Grad-CAM explanation. Install a compatible trained checkpoint to enable inference.</p></div>}
          <div className="result-grid">
            <article className="card"><span className="kicker">PREDICTED GRADE</span><h3>{result.predictedLabel||"Unavailable"}</h3><p className="muted">{demo?"Waiting for a trained classifier":"Model output — not a diagnosis"}</p>
              {!demo && <div className="probabilities">{result.probabilities?.map(p=><div className="prob" key={p.label}><div><span>{p.label}</span><b>{(p.probability*100).toFixed(1)}%</b></div><div className="track"><i style={{width:`${p.probability*100}%`}}/></div></div>)}</div>}
            </article>
            <article className="card"><span className="kicker">IMAGE QUALITY CHECK</span><h3 className="quality">{result.imageQuality?.status==="acceptable"?"No heuristic flags":"Review recommended"}</h3>
              {result.imageQuality?.issues?.length ? <ul>{result.imageQuality.issues.map((x,i)=><li key={i}>{x}</li>)}</ul>:<p className="muted">No issues detected by basic heuristic checks.</p>}
              <p className="muted tiny">{result.imageQuality?.note}</p>
            </article>
          </div>
          <article className="card heatmap"><div className="section-title"><h3>{result.heatmapAvailable?"Grad-CAM overlay":"Image preview"}</h3><span className="muted">{result.heatmapAvailable?"Attribution visualization":"No heatmap generated"}</span></div>
            {result.heatmapDataUrl && <img src={result.heatmapDataUrl} alt={result.heatmapAvailable?"Grad-CAM overlay":"Original image preview; not a heatmap"}/>}
            <p className="muted tiny">{result.heatmapAvailable?"Highlighted regions indicate areas influencing the model output; they do not prove a lesion.":"Shown only to verify image handling. This is not a model explanation."}</p>
          </article><div className="disclaimer">{result.disclaimer}</div>
        </section>}
      </section>
      <section className="panel workflow"><div className="section-title"><h2><small>03</small> How it works</h2></div><div className="steps">
        <article><span>01</span><h3>Preprocess</h3><p>Decode, resize and run simple image-quality heuristics.</p></article>
        <article><span>02</span><h3>Classify</h3><p>A trained CNN can estimate one of five DR severity classes.</p></article>
        <article><span>03</span><h3>Explain</h3><p>Grad-CAM visualizes regions influencing the model prediction.</p></article>
      </div></section>
    </main>
    <footer><span>RetinaLens · Explainable AI DR Screening Assistant</span><span>Educational research prototype · Not for clinical use</span></footer>
  </div>;
}
