export const API_BASE = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.hostname === "" || window.location.hostname.startsWith("192.168.") || window.location.hostname.startsWith("172."))
  ? `http://${window.location.hostname}:8000`
  : "https://urls-ssl-success-imported.trycloudflare.com";
export const API_TOKEN = "your_secure_api_token";
export const DEBUG_TOKEN = "your_secure_debug_token";
