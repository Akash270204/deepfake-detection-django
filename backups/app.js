// old js code 

// // DOM elements
// const uploadArea = document.getElementById('uploadArea');
// const fileInput = document.getElementById('fileInput');
// const uploadSection = document.getElementById('uploadSection');
// const previewSection = document.getElementById('previewSection');
// const preview = document.getElementById('preview');
// const analyzeBtn = document.getElementById('analyzeBtn');
// const resetBtn = document.getElementById('resetBtn');
// const loading = document.getElementById('loading');
// const results = document.getElementById('results');

// let uploadedFile = null;
// let uploadedFilename = null;

// // Click to upload
// uploadArea.addEventListener('click', () => {
//     fileInput.click();
// });

// // Drag and drop
// uploadArea.addEventListener('dragover', (e) => {
//     e.preventDefault();
//     uploadArea.style.background = 'rgba(102, 126, 234, 0.1)';
//     uploadArea.style.borderColor = '#764ba2';
// });

// uploadArea.addEventListener('dragleave', () => {
//     uploadArea.style.background = '';
//     uploadArea.style.borderColor = '#667eea';
// });

// uploadArea.addEventListener('drop', (e) => {
//     e.preventDefault();
//     uploadArea.style.background = '';
//     uploadArea.style.borderColor = '#667eea';
    
//     const file = e.dataTransfer.files[0];
//     if (file && file.type.startsWith('image/')) {
//         handleFile(file);
//     } else {
//         alert('Please upload an image file (JPG or PNG)');
//     }
// });

// // File input change
// fileInput.addEventListener('change', (e) => {
//     const file = e.target.files[0];
//     if (file) {
//         handleFile(file);
//     }
// });

// // Handle file
// function handleFile(file) {
//     uploadedFile = file;
    
//     const reader = new FileReader();
//     reader.onload = (e) => {
//         preview.src = e.target.result;
//         uploadSection.style.display = 'none';
//         previewSection.style.display = 'block';
//         results.style.display = 'none';
//     };
//     reader.readAsDataURL(file);
// }

// // Analyze button
// analyzeBtn.addEventListener('click', async () => {
//     if (!uploadedFile) return;
    
//     try {
//         analyzeBtn.disabled = true;
//         analyzeBtn.textContent = 'Analyzing...';
//         previewSection.style.display = 'none';
//         loading.style.display = 'block';
        
//         console.log('📤 Uploading file...');
        
//         // Upload file - FIXED: Added /api/ prefix
//         const formData = new FormData();
//         formData.append('image', uploadedFile);
        
//         const uploadRes = await fetch('/api/upload/', {  // ✅ Add /api/
//             method: 'POST',
//             body: formData
//         });
        
//         console.log('Upload response status:', uploadRes.status);
        
//         // Check if response is JSON
//         const contentType = uploadRes.headers.get('content-type');
//         if (!contentType || !contentType.includes('application/json')) {
//             const text = await uploadRes.text();
//             console.error('Non-JSON response:', text);
//             throw new Error('Server returned non-JSON response. Check server logs.');
//         }
        
//         const uploadData = await uploadRes.json();
//         console.log('Upload data:', uploadData);
        
//         if (!uploadData.success) {
//             throw new Error(uploadData.error || 'Upload failed');
//         }
        
//         uploadedFilename = uploadData.filename;
//         console.log('✅ File uploaded:', uploadedFilename);
        
//         console.log('🔍 Running prediction...');
        
//         // Predict - FIXED: Added /api/ prefix
//         const predictRes = await fetch('/api/predict/', {  // ✅ Add /api/
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify({ filename: uploadedFilename })
//         });
        
//         console.log('Predict response status:', predictRes.status);
        
//         // Check if response is JSON
//         const predictContentType = predictRes.headers.get('content-type');
//         if (!predictContentType || !predictContentType.includes('application/json')) {
//             const text = await predictRes.text();
//             console.error('Non-JSON response:', text);
//             throw new Error('Prediction failed. Check if model is trained.');
//         }
        
//         const predictData = await predictRes.json();
//         console.log('Prediction data:', predictData);
        
//         if (!predictData.success) {
//             throw new Error(predictData.error || 'Prediction failed');
//         }
        
//         console.log('✅ Prediction complete');
//         displayResults(predictData.result);
        
//     } catch (error) {
//         console.error('❌ Error:', error);
//         alert('Error: ' + error.message);
//         reset();
//     } finally {
//         analyzeBtn.disabled = false;
//         analyzeBtn.textContent = 'Analyze for Deepfake';
//         loading.style.display = 'none';
//     }
// });

// // Reset button
// resetBtn.addEventListener('click', reset);

// function reset() {
//     uploadedFile = null;
//     uploadedFilename = null;
//     fileInput.value = '';
//     uploadSection.style.display = 'block';
//     previewSection.style.display = 'none';
//     loading.style.display = 'none';
//     results.style.display = 'none';
// }

// function displayResults(result) {
//     const isDeepfake = result.isDeepfake;
    
//     // Heatmap HTML
//     let heatmapHTML = '';
//     if (result.heatmapPath && result.heatmapPath !== '') {
//         heatmapHTML = `
//             <div style="margin-top: 30px;">
//                 <h3 style="margin-bottom: 15px; color: #333;">🔥 AI Attention Heatmap</h3>
//                 <img src="/media/${result.heatmapPath}" 
//                      style="width: 100%; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
//                      alt="Heatmap">
//                 <p style="margin-top: 10px; font-size: 0.9rem; color: #666;">
//                     Red/yellow areas show where the AI model focused when making its decision.
//                 </p>
//             </div>
//         `;
//     }
    
//     // Facial analysis HTML
//     let facialHTML = '';
//     if (result.facialAnalysisPath && result.facialAnalysisPath !== '') {
//         facialHTML = `
//             <div style="margin-top: 30px;">
//                 <h3 style="margin-bottom: 15px; color: #333;">👤 Facial Region Analysis</h3>
//                 <img src="/media/${result.facialAnalysisPath}" 
//                      style="width: 100%; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
//                      alt="Facial Analysis">
//                 <p style="margin-top: 10px; font-size: 0.9rem; color: #666;">
//                     Analysis of facial regions for inconsistencies and manipulation artifacts.
//                 </p>
//             </div>
//         `;
//     }
    
//     results.innerHTML = `
//         <div class="result-card ${isDeepfake ? 'fake' : 'real'}">
//             <div class="result-header">
//                 <div class="result-icon">${isDeepfake ? '❌' : '✅'}</div>
//                 <div class="result-info">
//                     <h2>${isDeepfake ? 'Deepfake Detected' : 'Authentic Media'}</h2>
//                     <p class="confidence">Confidence: ${result.confidence}%</p>
//                 </div>
//             </div>
//             <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #666; margin-top: 10px;">
//                 <span>Model: ${result.modelUsed}</span>
//                 <span>Process Time: ${result.processTime}s</span>
//             </div>
//         </div>
        
//         ${heatmapHTML}
//         ${facialHTML}
        
//         <div class="indicators">
//             <h3>⚠️ Detection Indicators</h3>
//             ${result.indicators.map(ind => {
//                 const isAnomalous = ind.score > ind.threshold;
//                 return `
//                     <div class="indicator">
//                         <div class="indicator-header">
//                             <span><strong>${ind.name}</strong></span>
//                             <span style="font-weight: 600; color: ${isAnomalous ? '#ef4444' : '#34d399'}">
//                                 ${ind.score.toFixed(1)}% ${isAnomalous ? '⚠' : '✓'}
//                             </span>
//                         </div>
//                         <div class="indicator-bar">
//                             <div class="indicator-fill ${isAnomalous ? 'warning' : 'normal'}" 
//                                  style="width: ${ind.score}%"></div>
//                         </div>
//                     </div>
//                 `;
//             }).join('')}
//         </div>
        
//         ${isDeepfake ? `
//             <div class="warning-box">
//                 <span class="warning-icon">⚠️</span>
//                 <p class="warning-text">
//                     This media shows signs of digital manipulation. Consider verifying the source 
//                     and cross-referencing with other authentic sources before sharing.
//                 </p>
//             </div>
//         ` : ''}
//     `;
    
//     results.style.display = 'block';
// }
