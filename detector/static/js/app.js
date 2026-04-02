// ============================================================================
// DEEPFAKE DETECTOR - JAVASCRIPT
// Updated: added manipulation highlight section to video results
// ============================================================================

const uploadArea    = document.getElementById('uploadArea');
const fileInput     = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const imagePreview  = document.getElementById('preview');
const videoPreview  = document.getElementById('videoPreview');
const analyzeBtn    = document.getElementById('analyzeBtn');
const resetBtn      = document.getElementById('resetBtn');
const loading       = document.getElementById('loading');
const results       = document.getElementById('results');
const imageTabBtn   = document.getElementById('imageTabBtn');
const videoTabBtn   = document.getElementById('videoTabBtn');

let uploadedFile     = null;
let uploadedFilename = null;
let currentFileType  = 'image';

const isAuthenticated = document.body.dataset.authenticated === 'true';

// ============================================================================
// INIT
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
});

function initializeEventListeners() {
    uploadArea?.addEventListener('click', () => fileInput.click());
    uploadArea?.addEventListener('dragover', handleDragOver);
    uploadArea?.addEventListener('dragleave', handleDragLeave);
    uploadArea?.addEventListener('drop', handleDrop);
    fileInput?.addEventListener('change', handleFileInputChange);
    analyzeBtn?.addEventListener('click', handleAnalyze);
    resetBtn?.addEventListener('click', reset);
    imageTabBtn?.addEventListener('click', () => switchTab('image'));
    videoTabBtn?.addEventListener('click', () => switchTab('video'));
}

// ============================================================================
// TAB SWITCHING
// ============================================================================

function switchTab(type) {
    if (type === 'video' && !isAuthenticated) {
        window.location.href = '/signup/?video=true';
        return;
    }
    currentFileType = type;
    imageTabBtn?.classList.toggle('active', type === 'image');
    videoTabBtn?.classList.toggle('active', type === 'video');
    if (type === 'image') {
        fileInput?.setAttribute('accept', 'image/*');
        updateUploadAreaText('Image', 'JPG, PNG, JPEG', '50MB');
    } else {
        fileInput?.setAttribute('accept', 'video/*');
        updateUploadAreaText('Video', 'MP4, AVI, MOV, MKV, WebM', '200MB');
    }
    reset();
}

function updateUploadAreaText(mediaType, formats, maxSize) {
    const title    = uploadArea?.querySelector('h3');
    const subtitle = uploadArea?.querySelector('p');
    if (title)    title.textContent    = `Click to upload or drag and drop ${mediaType}`;
    if (subtitle) subtitle.textContent = `Supports: ${formats} (Max ${maxSize})`;
}

// ============================================================================
// DRAG AND DROP
// ============================================================================

function handleDragOver(e)  { e.preventDefault(); uploadArea.classList.add('dragover'); }
function handleDragLeave()  { uploadArea.classList.remove('dragover'); }
function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) validateAndHandleFile(file);
}
function handleFileInputChange(e) {
    const file = e.target.files[0];
    if (file) validateAndHandleFile(file);
}

// ============================================================================
// FILE VALIDATION
// ============================================================================

function validateAndHandleFile(file) {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');

    if (!isImage && !isVideo) { showError('Please upload an image or video file'); return; }

    if (isVideo && !isAuthenticated) {
        showAuthenticationRequired();
        setTimeout(() => { window.location.href = '/signup/?video=true'; }, 2500);
        return;
    }

    const maxSize = isImage ? 50 * 1024 * 1024 : 200 * 1024 * 1024;
    if (file.size > maxSize) {
        showError(`File too large. Maximum size is ${maxSize / (1024 * 1024)}MB`);
        return;
    }

    const allowedImage = ['image/jpeg', 'image/jpg', 'image/png'];
    const allowedVideo = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime',
                          'video/x-msvideo', 'video/x-matroska', 'video/webm'];

    if (isImage && !allowedImage.includes(file.type)) {
        showError('Invalid image format. Please use JPG or PNG'); return;
    }
    if (isVideo && !allowedVideo.includes(file.type)) {
        showError('Invalid video format. Please use MP4, AVI, MOV, MKV, or WebM'); return;
    }

    currentFileType = isImage ? 'image' : 'video';
    imageTabBtn?.classList.toggle('active', isImage);
    videoTabBtn?.classList.toggle('active', isVideo);
    handleFile(file);
}

function handleFile(file) {
    uploadedFile = file;
    const isVideo = file.type.startsWith('video/');

    const fileSizeIndicator = document.getElementById('fileSizeIndicator');
    const fileTypeIndicator = document.getElementById('fileTypeIndicator');
    if (fileSizeIndicator) fileSizeIndicator.textContent = formatFileSize(file.size);
    if (fileTypeIndicator) fileTypeIndicator.textContent = isVideo ? '🎬 Video' : '🖼️ Image';

    if (isVideo) {
        const videoURL = URL.createObjectURL(file);
        if (videoPreview) { videoPreview.src = videoURL; videoPreview.style.display = 'block'; }
        if (imagePreview) imagePreview.style.display = 'none';
    } else {
        const reader = new FileReader();
        reader.onload = (e) => {
            if (imagePreview) { imagePreview.src = e.target.result; imagePreview.style.display = 'block'; }
            if (videoPreview) videoPreview.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    if (uploadSection)   uploadSection.style.display = 'none';
    if (previewSection)  { previewSection.style.display = 'block'; previewSection.classList.add('scale-in'); }
    if (results)         results.style.display = 'none';
}

// ============================================================================
// ANALYZE
// ============================================================================

async function handleAnalyze() {
    if (!uploadedFile) return;
    try {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span>🔄</span> Analyzing...';
        if (previewSection) previewSection.style.display = 'none';
        if (loading)        loading.style.display = 'block';
        if (results)        results.style.display = 'none';

        updateLoadingText(currentFileType === 'video'
            ? 'Extracting & analyzing video frames...'
            : 'Analyzing image...');

        const formData = new FormData();
        formData.append(currentFileType, uploadedFile);

        const uploadRes  = await fetch('/upload/', { method: 'POST', body: formData });
        const uploadData = await handleResponse(uploadRes, 'upload');

        if (!uploadData.success) {
            if (uploadData.auth_required) { window.location.href = uploadData.redirect_url || '/signup/?video=true'; return; }
            throw new Error(uploadData.error || 'Upload failed');
        }

        uploadedFilename = uploadData.filename;
        const fileType   = uploadData.file_type || currentFileType;

        updateLoadingText(currentFileType === 'video'
            ? 'Running AI analysis on frames...'
            : 'Running AI analysis...');

        const predictRes  = await fetch('/predict/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: uploadedFilename, file_type: fileType })
        });
        const predictData = await handleResponse(predictRes, 'prediction');

        if (!predictData.success) {
            if (predictData.auth_required) { window.location.href = predictData.redirect_url || '/signup/?video=true'; return; }
            if (predictData.quality_issues) { showQualityError(predictData.quality_issues, predictData.metrics); return; }
            throw new Error(predictData.error || 'Prediction failed');
        }

        displayResults(predictData.result);

    } catch (error) {
        showError(error.message);
        if (uploadSection)  uploadSection.style.display = 'block';
        if (previewSection) previewSection.style.display = 'none';
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span>🔍</span> Analyze for Deepfake';
        if (loading) loading.style.display = 'none';
    }
}

async function handleResponse(response, type) {
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        throw new Error(`Server error during ${type}. Please check if the model is trained.`);
    }
    return response.json();
}

function updateLoadingText(text) {
    const loadingText = document.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = text;
}

// ============================================================================
// AUTH REQUIRED
// ============================================================================

function showAuthenticationRequired() {
    if (!results) return;
    results.innerHTML = `
        <div class="result-card" style="border:2px solid #f59e0b; background:linear-gradient(135deg,rgba(234,179,8,0.1),rgba(202,138,4,0.05));">
            <div class="result-header">
                <div class="result-icon">🔒</div>
                <div class="result-info">
                    <h2 style="color:#92400e;">Video Detection Requires Login</h2>
                    <p style="color:#78350f;margin-top:0.5rem;">Redirecting to sign-up page in 3 seconds…</p>
                </div>
            </div>
            <div style="padding:1.5rem;background:white;border-radius:12px;margin-top:1.5rem;">
                <h3 style="color:#667eea;margin-bottom:1rem;">🎬 Create Your Free Account</h3>
                <ul style="color:#666;line-height:2;margin-left:1.5rem;">
                    <li><strong>Analyze Videos</strong> — Upload and scan videos up to 200MB</li>
                    <li><strong>Frame-by-frame results</strong> — See every analysed frame</li>
                    <li><strong>Detection History</strong> — Track all your analyses</li>
                    <li><strong>100% Free</strong> — No credit card required</li>
                </ul>
                <div style="text-align:center;margin-top:1.5rem;display:flex;gap:1rem;justify-content:center;">
                    <a href="/signup/?video=true" class="btn btn-primary" style="text-decoration:none;padding:1rem 2rem;">✨ Sign Up Now</a>
                    <a href="/login/?video=true" class="btn btn-secondary" style="text-decoration:none;padding:1rem 2rem;">🔓 Already Have Account?</a>
                </div>
            </div>
        </div>`;
    results.style.display = 'block';
    results.classList.add('fade-in');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================================
// RESULTS DISPLAY
// ============================================================================

function displayResults(result) {
    if (!results) return;
    const isDeepfake = result.isDeepfake;
    const fileType   = result.file_type || currentFileType;

    let html = `
        <div class="result-card ${isDeepfake ? 'fake' : 'real'} scale-in">
            <div class="result-header">
                <div class="result-icon">${isDeepfake ? '❌' : '✅'}</div>
                <div class="result-info">
                    <h2>${isDeepfake ? '🚨 Deepfake Detected' : '✅ Authentic Media'}</h2>
                    <div class="confidence-badge">
                        <span>Confidence:</span>
                        <strong>${result.confidence}%</strong>
                    </div>
                </div>
            </div>
            <div class="probability-grid">
                <div class="probability-item">
                    <div class="probability-label">Real Probability</div>
                    <div class="probability-value real">${result.probabilities.real}%</div>
                </div>
                <div class="probability-item">
                    <div class="probability-label">Fake Probability</div>
                    <div class="probability-value fake">${result.probabilities.fake}%</div>
                </div>
                <div class="probability-item">
                    <div class="probability-label">Detection Threshold</div>
                    <div class="probability-value" style="color:var(--info)">${result.threshold}%</div>
                </div>
                <div class="probability-item">
                    <div class="probability-label">Model Used</div>
                    <div class="probability-value" style="color:var(--primary);font-size:1rem;">${result.modelUsed}</div>
                </div>
            </div>
        </div>`;

    if (result.rawConfidence !== undefined) {
        const margin       = Math.abs(result.rawConfidence - result.threshold).toFixed(2);
        const certainty    = result.decision?.certainty || 'moderate';
        const certaintyColor = certainty === 'high' ? '#10b981' : certainty === 'moderate' ? '#f59e0b' : '#ef4444';
        html += `
            <div class="model-analysis" style="margin-top:2rem;padding:1.5rem;background:rgba(0,0,0,0.03);border-radius:12px;">
                <h3 style="color:#667eea;margin-bottom:1rem;">🔬 Model Analysis Details</h3>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;">
                    <div style="padding:1rem;background:white;border-radius:8px;border-left:4px solid #667eea;">
                        <div style="font-size:0.85rem;color:#666;margin-bottom:0.5rem;">Raw Model Output</div>
                        <div style="font-size:1.75rem;font-weight:800;color:#667eea;">${result.rawConfidence.toFixed(2)}%</div>
                    </div>
                    <div style="padding:1rem;background:white;border-radius:8px;border-left:4px solid #f59e0b;">
                        <div style="font-size:0.85rem;color:#666;margin-bottom:0.5rem;">Detection Threshold</div>
                        <div style="font-size:1.75rem;font-weight:800;color:#f59e0b;">${result.threshold.toFixed ? result.threshold.toFixed(2) : result.threshold}%</div>
                    </div>
                    <div style="padding:1rem;background:white;border-radius:8px;border-left:4px solid ${certaintyColor};">
                        <div style="font-size:0.85rem;color:#666;margin-bottom:0.5rem;">Margin from Threshold</div>
                        <div style="font-size:1.75rem;font-weight:800;color:${certaintyColor};">±${margin}%</div>
                        <div style="font-size:0.8rem;color:#888;margin-top:0.25rem;text-transform:capitalize;">${certainty} certainty</div>
                    </div>
                </div>
                <div style="margin-top:1rem;padding:1rem;background:rgba(102,126,234,0.1);border-radius:8px;border-left:4px solid #667eea;">
                    <strong style="color:#333;">Decision Process:</strong><br>
                    <span style="color:#555;line-height:1.6;">${result.decision?.reason || `Model output ${result.rawConfidence.toFixed(2)}% compared to threshold ${result.threshold}%`}</span>
                </div>
            </div>`;
    }

    if (result.qualityMetrics && Object.keys(result.qualityMetrics).length > 0) {
        html += buildQualityMetrics(result.qualityMetrics, result.qualityWarnings);
    }

    if (fileType === 'video' && result.videoAnalysis) {
        html += buildVideoAnalysis(result.videoAnalysis);
    }

    if (fileType === 'image') {
        if (result.noFaceDetected) {
            html += `<div style="margin-top:1.5rem;padding:1rem;background:rgba(59,130,246,0.1);border-left:4px solid #3b82f6;border-radius:8px;">
                <strong>ℹ️ Note:</strong> No face detected. Showing overall image region analysis instead.</div>`;
        }
        if (result.heatmapPath)       html += buildVisualization('🔥 AI Attention Heatmap (Grad-CAM++)', result.heatmapPath, 'Red/yellow areas show where the AI focused. High-attention regions are most relevant for the classification decision.');
        if (result.facialAnalysisPath) html += buildVisualization(result.noFaceDetected ? '🗺️ Image Region Analysis' : '👤 Facial Region Analysis', result.facialAnalysisPath, result.noFaceDetected ? 'Image divided into regions and analysed for manipulation patterns.' : 'Detailed analysis of facial regions for inconsistencies.');
        if (result.artifactMapPath)    html += buildVisualization('🎨 AI Artifact Detection', result.artifactMapPath, 'Visual representation of AI-generated artifacts including edge inconsistencies and noise patterns.');
    }

    if (result.indicators && result.indicators.length > 0) {
        html += buildIndicators(result.indicators);
    }

    if (isDeepfake) {
        html += `
            <div class="warning-box slide-in-up">
                <span class="warning-icon">⚠️</span>
                <div class="warning-text">
                    <strong>Warning:</strong> This media shows signs of digital manipulation.
                    ${fileType === 'video' ? 'Video analysis detected manipulation in multiple frames.' : 'Consider verifying the source before sharing.'}
                    Always cross-reference with other authentic sources.
                </div>
            </div>`;
    }

    html += `<div style="margin-top:2rem;text-align:center;">
        <button onclick="reset()" class="btn btn-primary" style="max-width:300px;">
            🔄 Analyze Another ${fileType === 'video' ? 'Video' : 'Image'}
        </button></div>`;

    results.innerHTML = html;
    results.style.display = 'block';
    results.classList.add('fade-in');
    setTimeout(() => results.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

// ============================================================================
// BUILD HELPERS
// ============================================================================

function buildQualityMetrics(metrics, warnings) {
    let html = '<div class="quality-metrics fade-in"><h3 style="margin-bottom:1rem;">📊 Quality Metrics</h3><div class="quality-grid">';
    if (metrics.blur_score !== undefined) {
        const blurStatus = metrics.blur_score >= 100 ? '✅ Good' : metrics.blur_score >= 50 ? '⚠️ Fair' : '❌ Poor';
        html += `<div class="quality-item"><div class="quality-label">Blur Score</div><div class="quality-value">${Math.round(metrics.blur_score)}</div><div style="font-size:0.85rem;color:#666;margin-top:0.25rem;">${blurStatus}</div></div>`;
    }
    if (metrics.resolution) {
        html += `<div class="quality-item"><div class="quality-label">Resolution</div><div class="quality-value" style="font-size:1.1rem;">${metrics.resolution[0]}×${metrics.resolution[1]}</div></div>`;
    }
    html += '</div>';
    if (warnings && warnings.length > 0) {
        html += '<div style="margin-top:1rem;">';
        warnings.forEach(w => { html += `<div class="indicator warning" style="margin-bottom:0.5rem;"><div class="indicator-content"><span class="indicator-icon">⚠️</span><span class="indicator-text">${w}</span></div></div>`; });
        html += '</div>';
    }
    html += '</div>';
    return html;
}

function buildVideoAnalysis(videoData) {
    const tc   = videoData.temporalConsistency || {};
    const info = videoData.videoInfo || {};
    const ma   = videoData.manipulationAnalysis || {};   // NEW

    const consistencyIcon  = tc.is_consistent ? '✅' : '⚠️';
    const consistencyColor = tc.is_consistent ? 'var(--success)' : 'var(--warning)';

    let html = `
        <div class="video-timeline fade-in">
            <div class="timeline-header"><h3>🎬 Video Frame Analysis</h3></div>
            <div class="timeline-stats">
                <div class="timeline-stat"><div class="timeline-stat-value">${videoData.totalFrames}</div><div class="timeline-stat-label">Frames Analysed</div></div>
                <div class="timeline-stat"><div class="timeline-stat-value" style="color:var(--error)">${videoData.deepfakeFrames}</div><div class="timeline-stat-label">Deepfake Frames</div></div>
                <div class="timeline-stat"><div class="timeline-stat-value" style="color:${consistencyColor}">${consistencyIcon}</div><div class="timeline-stat-label">Temporal Consistency</div></div>
                <div class="timeline-stat"><div class="timeline-stat-value">${info.duration != null ? info.duration.toFixed(1) + 's' : 'N/A'}</div><div class="timeline-stat-label">Duration</div></div>
                <div class="timeline-stat"><div class="timeline-stat-value">${info.fps || 'N/A'} fps</div><div class="timeline-stat-label">Frame Rate</div></div>
                <div class="timeline-stat"><div class="timeline-stat-value" style="color:${videoData.deepfakeFrames > 0 ? 'var(--error)' : 'var(--success)'}">
                    ${videoData.deepfakePercentage != null ? videoData.deepfakePercentage.toFixed(1) : 0}%
                </div><div class="timeline-stat-label">Fake Frame %</div></div>
            </div>`;

    if (tc.message) {
        html += `<div style="padding:1rem;background:rgba(102,126,234,0.05);border-radius:8px;margin-top:1rem;">
            <strong>Consistency Analysis:</strong> ${tc.message}
            ${tc.variance != null ? `<br><small style="color:#666;">Variance: ${tc.variance.toFixed(2)}</small>` : ''}
        </div>`;
    }

    // ── NEW: Manipulation Highlight Section ───────────────────────────────────
    if (ma.has_manipulation) {
        html += buildManipulationHighlight(ma);
    }
    // ─────────────────────────────────────────────────────────────────────────

    if (videoData.frameByFrame && videoData.frameByFrame.length > 0) {
        html += buildFrameTimeline(videoData.frameByFrame);
    }

    html += '</div>';
    return html;
}

// ── NEW: Manipulation Highlight ───────────────────────────────────────────────
function buildManipulationHighlight(ma) {
    const locIcon = { beginning: '⏮️', middle: '⏯️', end: '⏭️' }[ma.manipulation_location] || '🎬';

    let html = `
        <div style="margin-top:1.5rem;padding:1.5rem;background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(220,38,38,0.04));
                    border:2px solid rgba(239,68,68,0.3);border-radius:12px;">
            <h4 style="color:#dc2626;margin-bottom:1rem;font-size:1.1rem;">
                🔴 Manipulation Detection Summary
            </h4>

            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1.5rem;">

                <div style="padding:1rem;background:white;border-radius:8px;border-left:4px solid #dc2626;">
                    <div style="font-size:0.8rem;color:#666;margin-bottom:0.25rem;">First fake frame detected at</div>
                    <div style="font-size:1.6rem;font-weight:800;color:#dc2626;">${ma.first_fake_timestamp.toFixed(1)}s</div>
                    <div style="font-size:0.75rem;color:#888;">Frame #${ma.first_fake_frame}</div>
                </div>

                <div style="padding:1rem;background:white;border-radius:8px;border-left:4px solid #f59e0b;">
                    <div style="font-size:0.8rem;color:#666;margin-bottom:0.25rem;">Manipulation location</div>
                    <div style="font-size:1.2rem;font-weight:800;color:#b45309;">${locIcon} ${ma.manipulation_location.charAt(0).toUpperCase() + ma.manipulation_location.slice(1)}</div>
                    <div style="font-size:0.75rem;color:#888;">of the video</div>
                </div>

                ${ma.longest_run_frame_count > 0 ? `
                <div style="padding:1rem;background:white;border-radius:8px;border-left:4px solid #dc2626;">
                    <div style="font-size:0.8rem;color:#666;margin-bottom:0.25rem;">Longest fake segment</div>
                    <div style="font-size:1.6rem;font-weight:800;color:#dc2626;">${ma.longest_run_frame_count} frames</div>
                    <div style="font-size:0.75rem;color:#888;">${ma.longest_run_start_timestamp.toFixed(1)}s – ${ma.longest_run_end_timestamp.toFixed(1)}s</div>
                </div>` : ''}
            </div>`;

    // First fake frame thumbnail
    if (ma.first_fake_thumbnail) {
        html += `
            <div style="margin-bottom:1.5rem;">
                <div style="font-size:0.85rem;font-weight:600;color:#dc2626;margin-bottom:0.5rem;">
                    📸 First manipulated frame — at ${ma.first_fake_timestamp.toFixed(1)}s
                    (${ma.first_fake_confidence.toFixed(1)}% fake probability)
                </div>
                <img src="${ma.first_fake_thumbnail}"
                     style="width:100%;max-width:360px;border-radius:8px;border:3px solid #dc2626;display:block;"
                     alt="First fake frame"
                     onerror="this.style.display='none'">
            </div>`;
    }

    // Top suspicious frames
    if (ma.top_suspicious_frames && ma.top_suspicious_frames.length > 0) {
        html += `
            <div>
                <div style="font-size:0.85rem;font-weight:600;color:#dc2626;margin-bottom:0.75rem;">
                    🔝 Most suspicious frames (highest fake probability)
                </div>
                <div style="display:flex;gap:0.75rem;flex-wrap:wrap;">`;

        ma.top_suspicious_frames.forEach((f, idx) => {
            const thumbHtml = f.thumbnail_url
                ? `<img src="${f.thumbnail_url}" alt="Frame ${f.frame_number}"
                        style="width:100%;height:80px;object-fit:cover;border-radius:6px 6px 0 0;"
                        onerror="this.style.display='none'">`
                : `<div style="height:80px;display:flex;align-items:center;justify-content:center;background:#fee2e2;border-radius:6px 6px 0 0;font-size:1.5rem;">🎞️</div>`;

            html += `
                <div style="width:140px;border:2px solid #dc2626;border-radius:8px;background:rgba(239,68,68,0.05);">
                    ${thumbHtml}
                    <div style="padding:0.4rem 0.5rem;">
                        <div style="font-size:0.7rem;color:#666;">Frame #${f.frame_number} @ ${f.timestamp.toFixed(1)}s</div>
                        <div style="font-size:0.85rem;font-weight:700;color:#dc2626;">${f.fake_probability.toFixed(1)}% fake</div>
                    </div>
                </div>`;
        });

        html += `</div></div>`;
    }

    html += '</div>';
    return html;
}

function buildFrameTimeline(frames) {
    const total     = frames.length;
    const fakeCount = frames.filter(f => f.is_deepfake).length;

    let html = `
        <div style="margin-top:1.5rem;">
            <h4 style="margin-bottom:0.5rem;">
                🎞️ Frame-by-Frame Detection
                <span style="font-size:0.85rem;font-weight:400;color:#666;margin-left:0.5rem;">
                    (${total} frames · ${fakeCount} fake · ${total - fakeCount} real)
                </span>
            </h4>
            <p style="color:#666;font-size:0.9rem;margin-bottom:1rem;">
                <span style="color:var(--error);">Red border = Fake</span> ·
                <span style="color:var(--success);">Green border = Real</span>
            </p>
            <div class="frame-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:0.75rem;">`;

    frames.forEach((frame) => {
        const isFake      = frame.is_deepfake;
        const borderColor = isFake ? 'var(--error)' : 'var(--success)';
        const bgColor     = isFake ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)';
        const icon        = isFake ? '❌' : '✅';
        const label       = isFake ? 'FAKE' : 'REAL';
        const labelColor  = isFake ? 'var(--error)' : 'var(--success)';
        const ts          = frame.timestamp != null ? frame.timestamp.toFixed(2) : '?';
        const conf        = frame.confidence != null ? frame.confidence.toFixed(1) : '?';

        const thumbHtml = frame.thumbnail_url
            ? `<img src="${frame.thumbnail_url}" alt="Frame ${frame.frame_number}"
                    style="width:100%;height:90px;object-fit:cover;border-radius:6px 6px 0 0;display:block;"
                    onerror="this.style.display='none';this.nextElementSibling.style.display='flex';">
               <div style="display:none;height:90px;align-items:center;justify-content:center;background:#f3f4f6;border-radius:6px 6px 0 0;font-size:2rem;">🎞️</div>`
            : `<div style="height:90px;display:flex;align-items:center;justify-content:center;background:#f3f4f6;border-radius:6px 6px 0 0;font-size:2rem;">🎞️</div>`;

        html += `
            <div style="border:2px solid ${borderColor};border-radius:8px;background:${bgColor};transition:transform 0.2s,box-shadow 0.2s;"
                 onmouseover="this.style.transform='scale(1.04)';this.style.boxShadow='0 4px 16px rgba(0,0,0,0.18)';"
                 onmouseout="this.style.transform='';this.style.boxShadow='';"
                 title="Frame #${frame.frame_number} @ ${ts}s — ${label} (${conf}%)">
                ${thumbHtml}
                <div style="padding:0.4rem 0.5rem;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="font-size:0.75rem;color:#555;font-weight:600;">#${frame.frame_number}</span>
                        <span style="font-size:0.75rem;color:#555;">${ts}s</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-top:0.2rem;">
                        <span style="font-size:0.8rem;font-weight:700;color:${labelColor};">${icon} ${label}</span>
                        <span style="font-size:0.75rem;color:#666;">${conf}%</span>
                    </div>
                </div>
            </div>`;
    });

    html += '</div></div>';
    return html;
}

function buildVisualization(title, imagePath, description) {
    return `
        <div class="visualization-section">
            <div class="visualization-header"><h3>${title}</h3></div>
            <img src="${imagePath}" class="visualization-image" alt="${title}" loading="lazy"
                 onerror="this.closest('.visualization-section').style.display='none'">
            <div class="visualization-description">${description}</div>
        </div>`;
}

function buildIndicators(indicators) {
    const iconMap = { error: '❌', warning: '⚠️', info: 'ℹ️', success: '✅' };
    let html = '<div class="indicators fade-in"><h3>⚠️ Detection Indicators</h3>';
    indicators.forEach((ind, i) => {
        const icon = iconMap[ind.type] || 'ℹ️';
        html += `<div class="indicator ${ind.type}" style="animation-delay:${i * 0.1}s">
            <div class="indicator-content">
                <span class="indicator-icon">${icon}</span>
                <span class="indicator-text">${ind.message}</span>
            </div></div>`;
    });
    html += '</div>';
    return html;
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    if (!results) return;
    results.innerHTML = `
        <div class="result-card" style="border:2px solid var(--error);background:rgba(239,68,68,0.1);">
            <div class="result-header">
                <div class="result-icon">❌</div>
                <div class="result-info"><h2>Error</h2></div>
            </div>
            <div style="padding:1.5rem;background:white;border-radius:8px;margin-top:1rem;">
                <p style="color:var(--error);font-size:1.1rem;line-height:1.6;">${message}</p>
            </div>
            <div style="margin-top:1.5rem;text-align:center;">
                <button onclick="reset()" class="btn btn-secondary">🔄 Try Again</button>
            </div>
        </div>`;
    results.style.display = 'block';
}

function showQualityError(issues, metrics) {
    if (!results) return;
    const issuesList = issues.map(i => `<li>${i}</li>`).join('');
    let metricsHTML  = '';
    if (metrics) {
        metricsHTML = '<div style="margin-top:1rem;padding:1rem;background:rgba(0,0,0,0.03);border-radius:8px;"><strong>Quality Metrics:</strong><ul style="margin-top:0.5rem;">';
        if (metrics.blur_score !== undefined) metricsHTML += `<li>Blur Score: ${metrics.blur_score.toFixed(1)} (minimum: 50)</li>`;
        if (metrics.resolution) metricsHTML += `<li>Resolution: ${metrics.resolution[0]}×${metrics.resolution[1]}</li>`;
        metricsHTML += '</ul></div>';
    }
    results.innerHTML = `
        <div class="result-card" style="border:2px solid var(--warning);background:rgba(234,179,8,0.1);">
            <div class="result-header">
                <div class="result-icon">⚠️</div>
                <div class="result-info"><h2>Quality Validation Failed</h2></div>
            </div>
            <div style="padding:1.5rem;background:white;border-radius:8px;margin-top:1rem;">
                <p style="margin-bottom:1rem;font-size:1.1rem;">The uploaded file does not meet quality requirements:</p>
                <ul style="color:var(--error);line-height:1.8;margin-left:1.5rem;">${issuesList}</ul>
                ${metricsHTML}
                <div style="margin-top:1.5rem;padding:1rem;background:rgba(59,130,246,0.1);border-left:4px solid var(--info);border-radius:4px;">
                    <strong>💡 Tip:</strong> Use a higher quality image with good lighting and resolution.
                </div>
            </div>
            <div style="margin-top:1.5rem;text-align:center;">
                <button onclick="reset()" class="btn btn-secondary">🔄 Upload Different File</button>
            </div>
        </div>`;
    results.style.display = 'block';
}

// ============================================================================
// RESET
// ============================================================================

function reset() {
    uploadedFile = null; uploadedFilename = null;
    if (fileInput) fileInput.value = '';
    if (uploadSection)  uploadSection.style.display = 'block';
    if (previewSection) { previewSection.style.display = 'none'; previewSection.classList.remove('scale-in'); }
    if (loading)  loading.style.display = 'none';
    if (results)  { results.style.display = 'none'; results.classList.remove('fade-in'); }
    if (imagePreview) imagePreview.src = '';
    if (videoPreview) { videoPreview.src = ''; videoPreview.pause(); }
}
window.reset = reset;

// ============================================================================
// UTILITY
// ============================================================================

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024, sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
}