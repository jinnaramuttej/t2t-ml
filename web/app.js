import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { API_BASE, API_TOKEN, DEBUG_TOKEN } from "./config.js";

const CONFIG = {
  // Hand bounds (fraction of frame area)
  handMinAreaFraction: 0.06,
  handMaxAreaFraction: 0.40,
  landmarkEdgeMargin: 0.02, // 2% inside every edge
  
  // Steadiness
  handSteadyMaxMoveFraction: 0.03, // of frame size over 1.0s
  handSteadyDurationMs: 1000, 
  frameSteadyMaxDiff: 5.0, 
  frameSteadyDurationMs: 1500,
  
  // Tapped bin circle
  binCircleRadiusFraction: 0.15,
  maxHandCoverFraction: 0.50, // hand must not cover more than 50%
  
  // Warnings
  tooDarkThreshold: 40, 
  blurryThreshold: 50,
};

let state = {
  phase: 0,
  sessionId: null,
  lat: null, lng: null,
  tapX: null, tapY: null,
  frame1: null, frame2: null,
  manual: false,
  handBox: null,
  handBoxState: false, // true=green, false=red
};

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const debugParams = new URLSearchParams(window.location.search);
const isDebug = debugParams.get('debug') === '1';

const elStep = document.getElementById('step-text');
const elTitle = document.getElementById('prompt-title');
const elSub = document.getElementById('prompt-secondary');

if (isDebug) document.getElementById('debug-controls').classList.remove('hidden');

function updateProgress(step) {
  for (let i = 1; i <= 4; i++) {
    const seg = document.getElementById('seg-' + i);
    if (i <= step) seg.classList.add('active');
    else seg.classList.remove('active');
  }
  elStep.innerText = `${step} / 4`;
}

function setPhase(p) {
  state.phase = p;
  document.querySelectorAll('.phase-ui').forEach(el => el.classList.add('hidden'));
  if (p >= 1 && p <= 4) {
    document.getElementById(`phase-${p}-ui`).classList.remove('hidden');
    updateProgress(p);
  }

  if (p === 1) {
    elTitle.innerText = "Point at the bin and tap it";
    elSub.innerText = "Keep the whole bin in view";
    overlay.onclick = handleTap;
  } else if (p === 2) {
    overlay.onclick = null;
    elTitle.innerText = "Hold the item up";
    elSub.innerText = "Keep the bin in view";
    startPhase2();
  } else if (p === 3) {
    elTitle.innerText = "Drop it in the bin";
    elSub.innerText = "Tap Done once it is in";
    startPhase3();
  } else if (p === 4) {
    elTitle.innerText = "Open palm above the bin";
    if (isDebug) document.getElementById('debug-gates').classList.remove('hidden');
    startPhase4();
  }
}

function handleTap(e) {
  if (state.phase !== 1) return;
  const rect = video.getBoundingClientRect();
  const vidRatio = video.videoWidth / video.videoHeight;
  const elRatio = rect.width / rect.height;
  let drawW, drawH, offsetX, offsetY;
  if (vidRatio > elRatio) {
    drawH = rect.height;
    drawW = drawH * vidRatio;
    offsetX = (rect.width - drawW) / 2;
    offsetY = 0;
  } else {
    drawW = rect.width;
    drawH = drawW / vidRatio;
    offsetX = 0;
    offsetY = (rect.height - drawH) / 2;
  }
  const tapX_px = e.clientX - rect.left - offsetX;
  const tapY_px = e.clientY - rect.top - offsetY;
  state.tapX = tapX_px / drawW;
  state.tapY = tapY_px / drawH;
  
  if (state.tapX >= 0 && state.tapX <= 1 && state.tapY >= 0 && state.tapY <= 1) {
    setPhase(2);
  }
}

function drawLoop() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  
  if (state.tapX !== null && state.tapY !== null) {
    const cx = state.tapX * overlay.width;
    const cy = state.tapY * overlay.height;
    ctx.strokeStyle = '#8FE388';
    
    if (state.phase === 1) {
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, cy, 48, 0, 2*Math.PI);
      ctx.stroke();
      ctx.fillStyle = '#8FE388';
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, 2*Math.PI);
      ctx.fill();
    } else {
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, 2*Math.PI);
      ctx.stroke();
    }
  }

  if (state.phase === 4 && state.handBox) {
    ctx.strokeStyle = state.handBoxState ? '#8FE388' : '#ff6b6b';
    ctx.lineWidth = 3;
    const {x, y, w, h} = state.handBox;
    const r = 18;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.stroke();
    
    if (state.p4Progress !== undefined) {
      const cx = x + w/2;
      const cy = y + h/2;
      const radius = Math.max(w, h)/2 + 12; 
      ctx.lineWidth = 8;
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, 2*Math.PI);
      ctx.stroke();
      
      ctx.strokeStyle = state.handBoxState ? '#8FE388' : '#ff6b6b';
      ctx.beginPath();
      ctx.arc(cx, cy, radius, -Math.PI/2, -Math.PI/2 + 2*Math.PI*state.p4Progress);
      ctx.stroke();
    }
  }

  requestAnimationFrame(drawLoop);
}

function getGrayscaleData(vid, width=32, height=32) {
  const c = document.createElement('canvas');
  c.width = width; c.height = height;
  const cctx = c.getContext('2d', {willReadFrequently: true});
  cctx.drawImage(vid, 0, 0, width, height);
  const imgData = cctx.getImageData(0, 0, width, height);
  const data = new Uint8Array(width * height);
  for(let i=0; i<data.length; i++) {
    data[i] = imgData.data[i*4]*0.299 + imgData.data[i*4+1]*0.587 + imgData.data[i*4+2]*0.114;
  }
  return data;
}

function laplacianVariance(vid) {
  const w = 256, h = 256;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const cctx = c.getContext('2d', {willReadFrequently: true});
  cctx.drawImage(vid, 0, 0, w, h);
  const imgData = cctx.getImageData(0, 0, w, h);
  const d = new Int16Array(w * h);
  for(let i=0; i<w*h; i++) {
    d[i] = imgData.data[i*4]*0.299 + imgData.data[i*4+1]*0.587 + imgData.data[i*4+2]*0.114;
  }
  let sum=0, sumSq=0, count=0;
  for(let y=1; y<h-1; y++){
    for(let x=1; x<w-1; x++){
      let idx = y*w+x;
      let val = d[idx]*4 - d[idx-1] - d[idx+1] - d[idx-w] - d[idx+w];
      sum += val; sumSq += val*val; count++;
    }
  }
  let mean = sum/count;
  return (sumSq/count) - (mean*mean);
}

async function captureCandidates(count, delayMs, callback) {
  let bestVar = -1;
  let bestBlob = null;
  for (let i=0; i<count; i++) {
    const canvas = document.createElement('canvas');
    let vw = video.videoWidth, vh = video.videoHeight;
    let scale = Math.min(720 / Math.max(vw, vh), 1.0);
    canvas.width = vw * scale;
    canvas.height = vh * scale;
    const cctx = canvas.getContext('2d');
    cctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    let variance = laplacianVariance(video);
    let blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.85));
    if (variance > bestVar) {
      bestVar = variance;
      bestBlob = blob;
    }
    if (delayMs > 0 && i < count - 1) {
      await new Promise(r => setTimeout(r, delayMs));
    }
  }
  callback(bestBlob);
}

let capturing = false;

function startPhase2() {
  let lastGray = null;
  let steadyStart = null;
  capturing = false;
  const pRing = document.getElementById('p2-ring-progress');
  
  function loop() {
    if (state.phase !== 2) return;
    const currentGray = getGrayscaleData(video);
    if (lastGray) {
      const diff = meanAbsDiff(lastGray, currentGray);
      if (diff < CONFIG.frameSteadyMaxDiff) {
        if (!steadyStart) steadyStart = Date.now();
      } else {
        steadyStart = null;
      }
    }
    lastGray = currentGray;
    
    let progress = 0;
    if (steadyStart && !capturing) {
      const elapsed = Date.now() - steadyStart;
      progress = Math.min(elapsed / CONFIG.frameSteadyDurationMs, 1);
      if (progress >= 1.0) {
        capturing = true;
        captureCandidates(3, 300, (bestBlob) => {
          state.frame1 = bestBlob;
          setPhase(3);
        });
      }
    }
    const circ = 402;
    pRing.style.strokeDashoffset = circ - (progress * circ);
    
    if (!capturing) requestAnimationFrame(loop);
  }
  
  function meanAbsDiff(d1, d2) {
    let sum = 0;
    for(let i=0; i<d1.length; i++) sum += Math.abs(d1[i] - d2[i]);
    return sum / d1.length;
  }
  
  loop();
}

function startPhase3() {
  const btnDone = document.getElementById('btn-done');
  btnDone.disabled = true;
  setTimeout(() => { btnDone.disabled = false; }, 1000);
  btnDone.onclick = () => {
    setPhase(4);
  };
}

let handLandmarker;
let phase4Start = 0;
let handCenterHistory = [];

function checkOpenHand(landmarks) {
  const w = video.videoWidth, h = video.videoHeight;
  const distPx = (lm1, lm2) => Math.hypot((lm1.x - lm2.x)*w, (lm1.y - lm2.y)*h);
  if (distPx(landmarks[0], landmarks[8]) <= distPx(landmarks[0], landmarks[6])) return false;
  if (distPx(landmarks[0], landmarks[12]) <= distPx(landmarks[0], landmarks[10])) return false;
  if (distPx(landmarks[0], landmarks[16]) <= distPx(landmarks[0], landmarks[14])) return false;
  if (distPx(landmarks[0], landmarks[20]) <= distPx(landmarks[0], landmarks[18])) return false;
  if (distPx(landmarks[17], landmarks[4]) <= distPx(landmarks[17], landmarks[3])) return false;
  return true;
}

function checkTapCoverage(box) {
  if (state.tapX == null || state.tapY == null) return false;
  const cx = state.tapX * video.videoWidth;
  const cy = state.tapY * video.videoHeight;
  const r = video.videoWidth * CONFIG.binCircleRadiusFraction;
  let pointsIn = 0, totalPts = 100;
  for(let i=0; i<totalPts; i++) {
    let px = cx + (Math.random()*2-1)*r;
    let py = cy + (Math.random()*2-1)*r;
    if (Math.hypot(px-cx, py-cy) <= r) {
      if (px >= box.x && px <= box.x+box.w && py >= box.y && py <= box.y+box.h) pointsIn++;
    }
  }
  return (pointsIn / (Math.PI/4 * totalPts)) > CONFIG.maxHandCoverFraction;
}

function updateChip(id, st) {
  if (!isDebug) return;
  const el = document.getElementById(id);
  el.className = 'chip ' + (st === 1 ? 'green' : (st === 0 ? 'amber' : 'red'));
}

function startPhase4() {
  phase4Start = Date.now();
  capturing = false;
  handCenterHistory = [];
  let steadyStart = null;
  
  document.getElementById('btn-manual-capture').onclick = () => {
    if (capturing) return;
    capturing = true;
    state.manual = true;
    captureCandidates(1, 0, (blob) => {
      state.frame2 = blob;
      upload();
    });
  };

  function loop() {
    if (state.phase !== 4) return;
    if (!handLandmarker) {
      requestAnimationFrame(loop);
      return;
    }
    
    let nowMs = performance.now();
    let results = handLandmarker.detectForVideo(video, nowMs);
    let w = video.videoWidth, h = video.videoHeight;
    
    let message = "";
    let allGatesGreen = false;
    
    if (Date.now() - phase4Start > 8000) {
      document.getElementById('btn-manual-capture').classList.remove('hidden');
    }

    if (results.landmarks && results.landmarks.length > 0) {
      let lms = results.landmarks[0];
      let minX = 1, minY = 1, maxX = 0, maxY = 0;
      lms.forEach(lm => {
        minX = Math.min(minX, lm.x); minY = Math.min(minY, lm.y);
        maxX = Math.max(maxX, lm.x); maxY = Math.max(maxY, lm.y);
      });
      
      let box = { x: minX*w, y: minY*h, w: (maxX-minX)*w, h: (maxY-minY)*h };
      state.handBox = box;
      
      let areaFrac = (box.w * box.h) / (w * h);
      let insideEdges = minX > CONFIG.landmarkEdgeMargin && minY > CONFIG.landmarkEdgeMargin && 
                        maxX < (1 - CONFIG.landmarkEdgeMargin) && maxY < (1 - CONFIG.landmarkEdgeMargin);
      let openHand = checkOpenHand(lms);
      
      let cx = minX + (maxX-minX)/2, cy = minY + (maxY-minY)/2;
      handCenterHistory.push({t: Date.now(), x: cx, y: cy});
      handCenterHistory = handCenterHistory.filter(pt => Date.now() - pt.t <= CONFIG.handSteadyDurationMs);
      
      let maxDist = 0;
      for(let pt of handCenterHistory) {
        let d = Math.hypot(pt.x - cx, pt.y - cy);
        if(d > maxDist) maxDist = d;
      }
      let steady = (maxDist <= CONFIG.handSteadyMaxMoveFraction * Math.max(w, h)) && handCenterHistory.length > 10;
      let overBin = checkTapCoverage(box);
      
      let sizeOk = (areaFrac >= CONFIG.handMinAreaFraction && areaFrac <= CONFIG.handMaxAreaFraction);
      updateChip('gate-size', sizeOk ? 1 : 0);
      updateChip('gate-inside', insideEdges ? 1 : 0);
      updateChip('gate-open', openHand ? 1 : 0);
      updateChip('gate-steady', steady ? 1 : 0);
      updateChip('gate-offbin', !overBin ? 1 : 0);
      
      if (areaFrac < CONFIG.handMinAreaFraction) message = "Move closer";
      else if (areaFrac > CONFIG.handMaxAreaFraction) message = "Move your hand back";
      else if (!insideEdges) message = "Show your whole hand";
      else if (!openHand) message = "Open your fingers";
      else if (!steady) message = "Hold still";
      else if (overBin) message = "Move your hand off the bin";
      else {
        message = "";
        allGatesGreen = true;
      }
    } else {
      state.handBox = null;
      message = "Hand not found";
      handCenterHistory = [];
      updateChip('gate-size', -1);
      updateChip('gate-inside', -1);
      updateChip('gate-open', -1);
      updateChip('gate-steady', -1);
      updateChip('gate-offbin', -1);
    }
    
    state.handBoxState = allGatesGreen;
    elSub.innerText = message;
    
    if (allGatesGreen) {
      if (!steadyStart) steadyStart = Date.now();
      state.p4Progress = Math.min((Date.now() - steadyStart) / CONFIG.handSteadyDurationMs, 1);
      if (state.p4Progress >= 1.0 && !capturing) {
        capturing = true;
        captureCandidates(3, 300, (bestBlob) => {
          state.frame2 = bestBlob;
          upload();
        });
      }
    } else {
      steadyStart = null;
      state.p4Progress = 0;
    }
    
    if (!capturing) requestAnimationFrame(loop);
  }
  loop();
}

async function upload() {
  document.getElementById('screen-loading').classList.remove('hidden');
  
  let formData = new FormData();
  formData.append('session_id', state.sessionId);
  formData.append('frame1', state.frame1, 'frame1.jpg');
  formData.append('frame2', state.frame2, 'frame2.jpg');
  if (state.tapX !== null) formData.append('tap_x', state.tapX);
  if (state.tapY !== null) formData.append('tap_y', state.tapY);
  formData.append('manual', state.manual);
  if (state.lat) formData.append('lat', state.lat);
  if (state.lng) formData.append('lng', state.lng);
  
  if (isDebug) {
    const testType = document.getElementById('test-type').value;
    if (testType) formData.append('test_type', testType);
  }
  formData.append('device', navigator.userAgent);
  
  let headers = { 'X-Api-Token': API_TOKEN };
  if (isDebug) headers['X-Debug-Token'] = DEBUG_TOKEN;
  
  try {
    const res = await fetch(`${API_BASE}/validate`, {
      method: 'POST',
      headers: headers,
      body: formData
    });
    const data = await res.json();
    showResult(data);
  } catch (err) {
    showResult({verdict: 'error', message: err.toString()});
  }
}

function showResult(data) {
  document.getElementById('screen-loading').classList.add('hidden');
  document.getElementById('screen-result').classList.remove('hidden');
  document.getElementById('camera-container').classList.add('hidden');
  
  const v = data.verdict;
  const resTitle = document.getElementById('res-title');
  const resSub = document.getElementById('res-subtitle');
  const resIcon = document.getElementById('res-icon-container');
  const btnPri = document.getElementById('btn-res-primary');
  
  resIcon.className = 'res-icon';
  let svg = '';
  
  if (v === 'approve') {
    resTitle.innerText = "Disposal verified";
    resSub.innerText = "Your submission was checked and saved.";
    svg = `<svg viewBox="0 0 24 24" fill="none" stroke="#0E1512" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" width="48" height="48"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    btnPri.innerText = "Finish";
    btnPri.onclick = () => location.reload();
  } else if (v === 'manual_review') {
    resTitle.innerText = "Sent for review";
    resSub.innerText = data.message;
    resIcon.classList.add('amber');
    svg = `<svg viewBox="0 0 24 24" fill="none" stroke="#0E1512" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" width="48" height="48"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>`;
    btnPri.innerText = "Finish";
    btnPri.onclick = () => location.reload();
  } else {
    resTitle.innerText = "Submission Rejected";
    resSub.innerText = data.message;
    resIcon.classList.add('red');
    svg = `<svg viewBox="0 0 24 24" fill="none" stroke="#0E1512" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" width="48" height="48"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
    btnPri.innerText = "Try again";
    btnPri.onclick = () => location.reload();
  }
  
  resIcon.innerHTML = svg;
  
  if (isDebug && data.debug) {
    const elDebug = document.getElementById('res-debug');
    elDebug.innerText = JSON.stringify(data.debug, null, 2);
    elDebug.classList.remove('hidden');
  }
  
  document.getElementById('btn-res-secondary').onclick = () => location.reload();
}

document.getElementById('btn-start').onclick = async () => {
  const btn = document.getElementById('btn-start');
  btn.disabled = true;
  btn.innerText = "Starting...";
  
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition((pos) => {
      state.lat = pos.coords.latitude;
      state.lng = pos.coords.longitude;
    }, () => {}, {timeout: 3000});
  }
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => {
      video.onloadedmetadata = () => {
        video.play();
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        resolve();
      };
    });
  } catch (err) {
    document.getElementById('screen-start').classList.add('hidden');
    document.getElementById('screen-error').classList.remove('hidden');
    return;
  }
  
  try {
    let headers = {'X-Api-Token': API_TOKEN};
    if (isDebug) headers['X-Debug-Token'] = DEBUG_TOKEN;
    const res = await fetch(`${API_BASE}/session`, { method: 'POST', headers });
    const data = await res.json();
    state.sessionId = data.session_id;
    
    document.getElementById('screen-start').classList.add('hidden');
    document.getElementById('camera-container').classList.remove('hidden');
    
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `hand_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 1
    });
    
    requestAnimationFrame(drawLoop);
    setPhase(1);
  } catch (err) {
    alert("Failed to start session: " + err.message);
    btn.disabled = false;
    btn.innerText = "Start";
  }
};

setInterval(() => {
  if(state.phase < 1 || state.phase > 4) return;
  const w = 64, h = 64;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const cctx = c.getContext('2d');
  cctx.drawImage(video, 0, 0, w, h);
  const imgData = cctx.getImageData(0, 0, w, h);
  let sum = 0;
  for(let i=0; i<w*h; i++) sum += imgData.data[i*4]*0.299 + imgData.data[i*4+1]*0.587 + imgData.data[i*4+2]*0.114;
  let avg = sum / (w*h);
  
  const elDark = document.getElementById('warn-dark');
  if (avg < CONFIG.tooDarkThreshold) elDark.classList.remove('hidden');
  else elDark.classList.add('hidden');
  
  let variance = laplacianVariance(video);
  const elBlurry = document.getElementById('warn-blurry');
  if (variance < CONFIG.blurryThreshold) elBlurry.classList.remove('hidden');
  else elBlurry.classList.add('hidden');
}, 500);

document.getElementById('btn-close').onclick = () => location.reload();
