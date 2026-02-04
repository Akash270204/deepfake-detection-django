// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

let uploadedFile = null;
let uploadedFilename = null;

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
    uploadArea.style.borderColor = '#764ba2';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = '';
    uploadArea.style.borderColor = '#667eea';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '';
    uploadArea.style.borderColor = '#667eea';
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        alert('Please upload an image file (JPG or PNG)');
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Handle file
function handleFile(file) {
    uploadedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        uploadSection.style.display = 'none';
        previewSection.style.display = 'block';
        results.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!uploadedFile) return;
    
    try {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        previewSection.style.display = 'none';
        loading.style.display = 'block';
        
        console.log('📤 Uploading file...');
        
        // Upload file
        const formData = new FormData();
        formData.append('image', uploadedFile);
        
        const uploadRes = await fetch('/upload/', {
            method: 'POST',
            body: formData
        });
        
        console.log('Upload response status:', uploadRes.status);
        
        const contentType = uploadRes.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await uploadRes.text();
            console.error('Non-JSON response:', text);
            throw new Error('Server returned non-JSON response. Check server logs.');
        }
        
        const uploadData = await uploadRes.json();
        console.log('Upload data:', uploadData);
        
        if (!uploadData.success) {
            throw new Error(uploadData.error || 'Upload failed');
        }
        
        uploadedFilename = uploadData.filename;
        console.log('✅ File uploaded:', uploadedFilename);
        
        console.log('🔍 Running prediction...');
        
        // Predict
        const predictRes = await fetch('/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename: uploadedFilename })
        });
        
        console.log('Predict response status:', predictRes.status);
        
        const predictContentType = predictRes.headers.get('content-type');
        if (!predictContentType || !predictContentType.includes('application/json')) {
            const text = await predictRes.text();
            console.error('Non-JSON response:', text);
            throw new Error('Prediction failed. Check if model is trained.');
        }
        
        const predictData = await predictRes.json();
        console.log('Prediction data:', predictData);
        
        if (!predictData.success) {
            throw new Error(predictData.error || 'Prediction failed');
        }
        
        console.log('✅ Prediction complete');
        displayResults(predictData.result);
        
    } catch (error) {
        console.error('❌ Error:', error);
        alert('Error: ' + error.message);
        reset();
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze for Deepfake';
        loading.style.display = 'none';
    }
});

// Reset button
resetBtn.addEventListener('click', reset);

function reset() {
    uploadedFile = null;
    uploadedFilename = null;
    fileInput.value = '';
    uploadSection.style.display = 'block';
    previewSection.style.display = 'none';
    loading.style.display = 'none';
    results.style.display = 'none';
}

function displayResults(result) {
    const isDeepfake = result.isDeepfake;
    
    // Build heatmap HTML
    let heatmapHTML = '';
    if (result.heatmapPath) {
        heatmapHTML = `
            <div style="margin-top: 30px;">
                <h3 style="margin-bottom: 15px; color: #333; display: flex; align-items: center; gap: 10px;">
                    🔥 AI Attention Heatmap
                </h3>
                <img src="${result.heatmapPath}" 
                     style="width: 100%; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
                     alt="Heatmap"
                     onerror="this.style.display='none'; console.error('Heatmap failed to load');">
                <p style="margin-top: 10px; font-size: 0.9rem; color: #666;">
                    Red/yellow areas show where the AI model focused when making its decision.
                </p>
            </div>
        `;
    }
    
    // Build facial analysis HTML
    let facialHTML = '';
    if (result.facialAnalysisPath) {
        facialHTML = `
            <div style="margin-top: 30px;">
                <h3 style="margin-bottom: 15px; color: #333; display: flex; align-items: center; gap: 10px;">
                    👤 Facial Region Analysis
                </h3>
                <img src="${result.facialAnalysisPath}" 
                     style="width: 100%; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
                     alt="Facial Analysis"
                     onerror="this.style.display='none'; console.error('Facial analysis failed to load');">
                <p style="margin-top: 10px; font-size: 0.9rem; color: #666;">
                    Analysis of facial regions for inconsistencies and manipulation artifacts.
                </p>
            </div>
        `;
    }
    
    // Build indicators HTML
    let indicatorsHTML = '';
    if (result.indicators && result.indicators.length > 0) {
        const indicatorItems = result.indicators.map(ind => {
            const iconMap = {
                'error': '❌',
                'warning': '⚠️',
                'info': 'ℹ️',
                'success': '✅'
            };
            const colorMap = {
                'error': '#ef4444',
                'warning': '#f59e0b',
                'info': '#3b82f6',
                'success': '#10b981'
            };
            const icon = iconMap[ind.type] || 'ℹ️';
            const color = colorMap[ind.type] || '#3b82f6';
            
            return `
                <div class="indicator" style="padding: 12px; margin-bottom: 8px; border-left: 3px solid ${color}; background: rgba(0,0,0,0.03); border-radius: 6px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.2em;">${icon}</span>
                        <span style="color: #333;">${ind.message}</span>
                    </div>
                </div>
            `;
        }).join('');
        
        indicatorsHTML = `
            <div class="indicators" style="margin-top: 30px;">
                <h3 style="margin-bottom: 15px; color: #333;">⚠️ Detection Indicators</h3>
                ${indicatorItems}
            </div>
        `;
    }
    
    results.innerHTML = `
        <div class="result-card ${isDeepfake ? 'fake' : 'real'}">
            <div class="result-header">
                <div class="result-icon">${isDeepfake ? '❌' : '✅'}</div>
                <div class="result-info">
                    <h2>${isDeepfake ? 'Deepfake Detected' : 'Authentic Media'}</h2>
                    <p class="confidence">Confidence: ${result.confidence}%</p>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.03); border-radius: 8px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 0.9rem;">
                    <div>
                        <div style="color: #666; margin-bottom: 5px;">Real Probability</div>
                        <div style="font-size: 1.2em; font-weight: 600; color: #10b981;">
                            ${result.probabilities.real}%
                        </div>
                    </div>
                    <div>
                        <div style="color: #666; margin-bottom: 5px;">Fake Probability</div>
                        <div style="font-size: 1.2em; font-weight: 600; color: #ef4444;">
                            ${result.probabilities.fake}%
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #666; margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                <span>Model: ${result.modelUsed}</span>
                <span>Process Time: ${result.processTime}s</span>
            </div>
        </div>
        
        ${heatmapHTML}
        ${facialHTML}
        ${indicatorsHTML}
        
        ${isDeepfake ? `
            <div class="warning-box" style="margin-top: 20px; padding: 15px; background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 8px;">
                <div style="display: flex; gap: 12px; align-items: start;">
                    <span class="warning-icon" style="font-size: 1.5em;">⚠️</span>
                    <p class="warning-text" style="margin: 0; color: #991b1b; line-height: 1.6;">
                        This media shows signs of digital manipulation. Consider verifying the source 
                        and cross-referencing with other authentic sources before sharing.
                    </p>
                </div>
            </div>
        ` : ''}
        
        <div style="margin-top: 20px; text-align: center;">
            <button onclick="reset()" style="padding: 10px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; font-weight: 600;">
                Analyze Another Image
            </button>
        </div>
    `;
    
    results.style.display = 'block';
    
    // Log for debugging
    console.log('✅ Results displayed successfully');
    console.log('Result data:', result);
}

// Make reset function global
window.reset = reset;

console.log('✅ app.js loaded successfully');