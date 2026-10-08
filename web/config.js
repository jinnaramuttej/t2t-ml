export const API_BASE = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? "http://localhost:8000"
  : "https://ingredients-mtv-models-cork.trycloudflare.com";
export const API_TOKEN = "your_secure_api_token";
export const DEBUG_TOKEN = "your_secure_debug_token";
