
import { useState } from "react";
import axios from "axios";
import "./Login.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:4000";

export default function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  
function handleSubmit(e) {
  e.preventDefault();
  setError("");

  if (!email.trim() || !password) {
    setError("Please enter your email and password.");
    return;
  }

  // Temporary demo login for presentation
  onLogin("presentation-demo");
}

  return (
    <div className="auth-page">
      <div className="auth-card">
        <a className="auth-brand" href="#">
          <span className="auth-logo">R</span>
          Retina<span>Lens</span>
        </a>

        <div className="auth-heading">
          <div className="auth-eyebrow">EXPLAINABLE AI · RETINAL SCREENING</div>
          <h1>Welcome back.</h1>
          <p>Sign in to your RetinaLens screening workspace.</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="auth-field">
            <label htmlFor="login-email">Email address</label>
            <input
              id="login-email"
              type="email"
              placeholder="you@example.com"
              autoComplete="username"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="auth-field">
            <label htmlFor="login-password">Password</label>
            <div className="auth-password">
              <input
                id="login-password"
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                className="auth-show"
                onClick={() => setShowPassword((value) => !value)}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </div>

          {error && (
            <p className="auth-error" role="alert">
              {error}
            </p>
          )}

          <button className="auth-submit" type="submit" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
            {!loading && <span>→</span>}
          </button>
        </form>

        <div className="auth-note">
          <span>✳</span>
          <p>
            Designed for retinal image research and explainable AI.
          </p>
        </div>

        <p className="auth-disclaimer">
          Educational research prototype only. Not for diagnosis or
          treatment decisions. All screening outputs require qualified
          professional review.
        </p>
      </div>
    </div>
  );
}
