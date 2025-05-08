/**
 * Main JavaScript file for the AI-Driven Structural Health Monitoring web interface.
 */

// DOM Elements - Analysis Tab
const uploadAreaBtn = document.getElementById('uploadBtn');
const cameraBtn = document.getElementById('cameraBtn');
const droneSimBtn = document.getElementById('droneSimBtn');
const uploadArea = document.getElementById('uploadArea');
const cameraArea = document.getElementById('cameraArea');
const droneSimArea = document.getElementById('droneSimArea');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const videoElement = document.getElementById('videoElement');
const canvasElement = document.getElementById('canvasElement');
const captureBtn = document.getElementById('captureBtn');
const originalImage = document.getElementById('originalImage');
const droneImage = document.getElementById('droneImage');
const altitudeSlider = document.getElementById('altitudeSlider');
const altitudeValue = document.getElementById('altitudeValue');
const angleSlider = document.getElementById('angleSlider');
const angleValue = document.getElementById('angleValue');
const rotationSlider = document.getElementById('rotationSlider');
const rotationValue = document.getElementById('rotationValue');
const weatherSelector = document.getElementById('weatherSelector');
const randomDroneView = document.getElementById('randomDroneView');
const analyzeDroneView = document.getElementById('analyzeDroneView');

// DOM Elements - Results
const predictionCard = document.getElementById('predictionCard');
const resultIcon = document.getElementById('resultIcon');
const predictionResult = document.getElementById('predictionResult');
const predictionSubtext = document.getElementById('predictionSubtext');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const probabilitiesList = document.getElementById('probabilitiesList');
const defectDetails = document.getElementById('defectDetails');
const defectType = document.getElementById('defectType');
const defectSeverity = document.getElementById('defectSeverity');
const defectSize = document.getElementById('defectSize');
const defectLocation = document.getElementById('defectLocation');
const saveResultBtn = document.getElementById('saveResultBtn');
const addToReportBtn = document.getElementById('addToReportBtn');

// DOM Elements - Monitoring Tab
const structureSelector = document.getElementById('structureSelector');
const viewAllBtn = document.getElementById('viewAllBtn');
const scheduleDroneBtn = document.getElementById('scheduleDroneBtn');
const structureViewer = document.getElementById('structureViewer');
const sensorSelect = document.getElementById('sensorSelect');
const timeRangeSelect = document.getElementById('timeRangeSelect');
const sensorChart = document.getElementById('sensorChart');

// Loading and Error Elements
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// Chart instances
let sensorChartInstance = null;

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Analysis Tab features
    initImageSourceTabs();
    initDropzone();
    initCamera();
    initDroneSimulation();
    initResultActions();
    
    // Initialize Monitoring Tab features
    initStructureViewer();
    initSensorCharts();
    initAlerts();
    
    // Initialize Dashboard features
    
    // Hide initial elements
    hideLoading();
    hideError();
    
    // Initialize tooltips and other Bootstrap components
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Update last update time
    updateLastUpdateTime();
});

/**
 * Initialize image source tabs in the Analysis section
 */
function initImageSourceTabs() {
    // Only proceed if elements exist
    if (!uploadAreaBtn || !cameraBtn || !droneSimBtn) return;
    
    // Initially hide camera area and drone sim area
    if (cameraArea) cameraArea.classList.add('d-none');
    if (droneSimArea) droneSimArea.classList.add('d-none');
    
    // Upload button
    uploadAreaBtn.addEventListener('click', () => {
        setActiveTab(uploadAreaBtn);
        if (uploadArea) uploadArea.classList.remove('d-none');
        if (cameraArea) cameraArea.classList.add('d-none');
        if (droneSimArea) droneSimArea.classList.add('d-none');
    });
    
    // Camera button
    cameraBtn.addEventListener('click', () => {
        setActiveTab(cameraBtn);
        if (uploadArea) uploadArea.classList.add('d-none');
        if (cameraArea) cameraArea.classList.remove('d-none');
        if (droneSimArea) droneSimArea.classList.add('d-none');
        initCamera(); // Make sure camera is initialized
    });
    
    // Drone Sim button
    droneSimBtn.addEventListener('click', () => {
        setActiveTab(droneSimBtn);
        if (uploadArea) uploadArea.classList.add('d-none');
        if (cameraArea) cameraArea.classList.add('d-none');
        if (droneSimArea) droneSimArea.classList.remove('d-none');
    });
}

/**
 * Set active tab
 */
function setActiveTab(activeBtn) {
    // Remove active class from all buttons
    uploadAreaBtn.classList.remove('active');
    cameraBtn.classList.remove('active');
    droneSimBtn.classList.remove('active');
    
    // Add active class to selected button
    activeBtn.classList.add('active');
}

/**
 * Initialize dropzone for file uploads
 */
function initDropzone() {
    if (!dropzone || !fileInput) return;
    
    // Click to select file
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop events
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('active');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('active');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

/**
 * Handle file upload
 * @param {File} file - The uploaded file
 */
function handleFile(file) {
    // Check if file is an image
    if (!file.type.match('image.*')) {
        showError('Please upload an image file.');
        return;
    }

    // Read and preview file
    const reader = new FileReader();
    reader.onload = (e) => {
        if (imagePreview) {
        imagePreview.src = e.target.result;
            imagePreview.classList.remove('d-none');
        }
        
        // Also set original image in drone simulation
        if (originalImage) {
            originalImage.src = e.target.result;
        }
        
        uploadImage(file);
    };
    reader.readAsDataURL(file);
}

/**
 * Upload image to server for prediction
 * @param {File} file - The image file to upload
 */
function uploadImage(file) {
    // Show loading
    showLoading('Analyzing image...');
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Send to server
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Server error');
        }
        return response.json();
    })
    .then(data => {
        // Hide loading
        hideLoading();
        
        // Display prediction
        displayPrediction(data);
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoading();
        showError('Error analyzing image. Please try again.');
    });
}

/**
 * Display prediction results
 * @param {Object} data - Prediction data from server
 */
function displayPrediction(data) {
    // Show prediction card
    if (predictionCard) {
    predictionCard.style.display = 'block';
    }
    
    const isProblem = data.prediction === 'positive';
    
    // Set icon
    if (resultIcon) {
        resultIcon.className = isProblem 
            ? 'fas fa-exclamation-circle fa-3x text-danger' 
            : 'fas fa-check-circle fa-3x text-success';
    }
    
    // Set prediction result text
    if (predictionResult) {
        predictionResult.textContent = isProblem ? 'DEFECT DETECTED' : 'NO DEFECT DETECTED';
        predictionResult.className = isProblem ? 'mb-1 text-danger' : 'mb-1 text-success';
    }
    
    // Set prediction subtext
    if (predictionSubtext) {
        predictionSubtext.textContent = isProblem 
            ? 'Structural defect found - inspection recommended' 
            : 'Structure appears to be in good condition';
    }
    
    // Set confidence bar
    const confidence = data.confidence * 100;
    if (confidenceBar) {
    confidenceBar.style.width = `${confidence}%`;
    confidenceBar.className = isProblem ? 'progress-bar bg-danger' : 'progress-bar bg-success';
    }
    
    if (confidenceText) {
        confidenceText.textContent = `${confidence.toFixed(1)}%`;
    }
    
    // Set detailed probabilities
    if (probabilitiesList) {
    probabilitiesList.innerHTML = '';
    if (data.probabilities) {
        Object.entries(data.probabilities).forEach(([label, prob]) => {
            const item = document.createElement('li');
            item.className = 'list-group-item d-flex justify-content-between align-items-center';
            item.innerHTML = `
                ${label}
                <span class="badge ${prob >= 0.5 ? 'bg-primary' : 'bg-secondary'} rounded-pill">
                        ${(prob * 100).toFixed(1)}%
                </span>
            `;
            probabilitiesList.appendChild(item);
        });
        }
    }
    
    // Show defect details if a defect is detected
    if (defectDetails) {
        if (isProblem) {
            defectDetails.classList.remove('d-none');
            // In a real app, these would come from the server with the actual analysis
            // Here we're just setting example values
            if (defectType) defectType.textContent = 'Crack';
            if (defectSeverity) defectSeverity.textContent = 'Medium';
            if (defectSize) defectSize.textContent = '12.5 cm';
            if (defectLocation) defectLocation.textContent = 'Upper right';
        } else {
            defectDetails.classList.add('d-none');
        }
    }
    
    // Add to recent analyses
    addToRecentAnalyses(data);
}

/**
 * Add analysis to recent analyses list
 */
function addToRecentAnalyses(data) {
    const recentAnalysesList = document.getElementById('recentAnalysesList');
    if (!recentAnalysesList) return;
    
    // Get the image source
    const imgSrc = imagePreview ? imagePreview.src : '#';
    const isProblem = data.prediction === 'positive';
    
    // Create new entry
    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td>Just now</td>
        <td><img src="${imgSrc}" alt="thumbnail" width="40" height="40" class="rounded"></td>
        <td><span class="badge ${isProblem ? 'bg-danger' : 'bg-success'}">${isProblem ? 'Defect' : 'No Defect'}</span></td>
        <td><button class="btn btn-sm btn-outline-primary"><i class="fas fa-eye"></i></button></td>
    `;
    
    // Add to top of list
    if (recentAnalysesList.firstChild) {
        recentAnalysesList.insertBefore(tr, recentAnalysesList.firstChild);
    } else {
        recentAnalysesList.appendChild(tr);
    }
    
    // Remove oldest entry if more than 5
    if (recentAnalysesList.children.length > 5) {
        recentAnalysesList.removeChild(recentAnalysesList.lastChild);
    }
}

/**
 * Initialize camera functionality
 */
function initCamera() {
    if (!videoElement || !captureBtn) return;
    
    captureBtn.disabled = true;
    
    // Check if camera is available
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        captureBtn.disabled = false;
        
        // Only start camera when the camera area is visible
        if (cameraArea && !cameraArea.classList.contains('d-none')) {
            // Access camera
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    videoElement.srcObject = stream;
                    videoElement.play();
                })
                .catch(err => {
                    console.error('Error accessing camera:', err);
                    showError('Could not access camera. Please check permissions.');
                });
        }
    } else {
        showError('Camera not available on this device or browser.');
    }
    
    // Capture button
    captureBtn.addEventListener('click', capturePhoto);
}

/**
 * Capture photo from camera
 */
function capturePhoto() {
    if (!videoElement || !canvasElement) return;
    
    const context = canvasElement.getContext('2d');
    
    // Set canvas dimensions to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    // Convert to file
    canvasElement.toBlob(blob => {
        const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
        
        // Show preview
        if (imagePreview) {
            imagePreview.src = URL.createObjectURL(blob);
            imagePreview.classList.remove('d-none');
        }
        
        // Also set original image in drone simulation
        if (originalImage) {
            originalImage.src = URL.createObjectURL(blob);
        }
        
        // Upload for analysis
        uploadImage(file);
    }, 'image/jpeg', 0.9);
}

/**
 * Initialize drone simulation features
 */
function initDroneSimulation() {
    if (!altitudeSlider || !angleSlider || !rotationSlider || !weatherSelector || !randomDroneView || !analyzeDroneView) return;
    
    // Update values when sliders change
    altitudeSlider.addEventListener('input', updateDroneSimulation);
    angleSlider.addEventListener('input', updateDroneSimulation);
    rotationSlider.addEventListener('input', updateDroneSimulation);
    weatherSelector.addEventListener('change', updateDroneSimulation);
    
    // Random drone view button
    randomDroneView.addEventListener('click', () => {
        // Set random values for drone simulation
        altitudeSlider.value = Math.floor(Math.random() * 90) + 10; // 10-100
        angleSlider.value = Math.floor(Math.random() * 90); // 0-90
        rotationSlider.value = Math.floor(Math.random() * 90) - 45; // -45 to 45
        
        // Random weather condition
        const weatherOptions = ['normal', 'sunny', 'cloudy', 'rain'];
        weatherSelector.value = weatherOptions[Math.floor(Math.random() * weatherOptions.length)];
        
        updateDroneSimulation();
    });
    
    // Analyze drone view button
    analyzeDroneView.addEventListener('click', () => {
        if (!droneImage || !droneImage.src) {
            showError('Please upload an image first to create a drone view.');
            return;
        }
        
        // Get the drone view image and convert to a file for analysis
        fetch(droneImage.src)
            .then(res => res.blob())
            .then(blob => {
                const file = new File([blob], "drone-view.jpg", { type: "image/jpeg" });
                uploadImage(file);
            })
            .catch(err => {
                console.error('Error converting drone image:', err);
                showError('Could not analyze drone view. Please try again.');
            });
    });
}

/**
 * Update drone simulation based on slider values
 */
function updateDroneSimulation() {
    if (!originalImage || !originalImage.src || !droneImage) return;
    
    // Update display values
    if (altitudeValue) altitudeValue.textContent = `${altitudeSlider.value}m`;
    if (angleValue) angleValue.textContent = `${angleSlider.value}°`;
    if (rotationValue) rotationValue.textContent = `${rotationSlider.value}°`;
    
    // Get parameters
    const altitude = parseInt(altitudeSlider.value);
    const angle = parseInt(angleSlider.value);
    const rotation = parseInt(rotationSlider.value);
    const weather = weatherSelector.value;
    
    // Create form data for drone simulation API
    const formData = new FormData();
    
    // Convert original image to blob
    fetch(originalImage.src)
        .then(res => res.blob())
        .then(blob => {
            formData.append('file', blob, 'original.jpg');
            formData.append('perspective', (angle / 90).toFixed(2));
            formData.append('rotation', (rotation / 45).toFixed(2));
            
            // Send to server for simulation
            return fetch('/simulate_drone', {
                method: 'POST',
                body: formData
            });
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error');
            }
            return response.json();
        })
        .then(data => {
            // Display simulated drone image
            droneImage.src = `data:image/jpeg;base64,${data.image}`;
        })
        .catch(error => {
            console.error('Error simulating drone view:', error);
            // For demo purposes, just display the original with a filter applied
            applyDroneEffect(originalImage, droneImage, angle, rotation, weather);
        });
}

/**
 * Apply a simple drone effect to the image (client-side fallback)
 * In a real application, this would be done on the server with more sophisticated algorithms
 */
function applyDroneEffect(originalImg, targetImg, angle, rotation, weather) {
    // Create a canvas to apply effects
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set dimensions
    canvas.width = originalImg.naturalWidth || 500;
    canvas.height = originalImg.naturalHeight || 500;
    
    // Draw original image
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(rotation * Math.PI / 180);
    
    // Apply perspective (simple effect)
    const scale = 1.0 - (angle / 200); // Smaller scale for higher angles
    ctx.scale(1, scale);
    
    // Draw the image
    ctx.drawImage(originalImg, -canvas.width / 2, -canvas.height / 2, canvas.width, canvas.height);
    ctx.restore();
    
    // Apply weather effects (simple filters)
    if (weather === 'sunny') {
        // High contrast, warm tint
        ctx.globalCompositeOperation = 'multiply';
        ctx.fillStyle = 'rgba(255, 220, 150, 0.2)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else if (weather === 'cloudy') {
        // Lower contrast, bluish tint
        ctx.globalCompositeOperation = 'screen';
        ctx.fillStyle = 'rgba(150, 150, 180, 0.2)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else if (weather === 'rain') {
        // Add rain effect
        ctx.globalCompositeOperation = 'screen';
        ctx.fillStyle = 'rgba(100, 100, 150, 0.3)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add some random streaks
        ctx.strokeStyle = 'rgba(200, 200, 255, 0.3)';
        ctx.lineWidth = 1;
        for (let i = 0; i < 100; i++) {
            const x = Math.random() * canvas.width;
            const y = Math.random() * canvas.height;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + 5, y + 15);
            ctx.stroke();
        }
    }
    
    // Add drone camera overlay
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.lineWidth = 2;
    
    // Crosshair
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    ctx.beginPath();
    ctx.moveTo(centerX - 20, centerY);
    ctx.lineTo(centerX + 20, centerY);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - 20);
    ctx.lineTo(centerX, centerY + 20);
    ctx.stroke();
    
    // Set the result as the target image src
    targetImg.src = canvas.toDataURL('image/jpeg');
}

/**
 * Initialize result action buttons
 */
function initResultActions() {
    if (!saveResultBtn || !addToReportBtn) return;
    
    saveResultBtn.addEventListener('click', () => {
        // In a real app, this would save the result to the server
        alert('Result saved successfully!');
    });
    
    addToReportBtn.addEventListener('click', () => {
        // In a real app, this would add the result to a report
        alert('Result added to report!');
    });
}

/**
 * Initialize structure viewer in monitoring tab
 */
function initStructureViewer() {
    if (!structureViewer || !structureSelector) return;
    
    // In a real app, this would initialize a 3D viewer using Three.js
    // For now, just change the structure name on selection
    structureSelector.addEventListener('change', () => {
        const structureName = structureSelector.options[structureSelector.selectedIndex].text;
        const overlay = structureViewer.querySelector('.structure-overlay h6');
        if (overlay) {
            overlay.textContent = structureName;
        }
    });
    
    // Handle structure view buttons
    if (viewAllBtn) {
        viewAllBtn.addEventListener('click', () => {
            alert('Viewing all structures is not implemented in this demo.');
        });
    }
    
    if (scheduleDroneBtn) {
        scheduleDroneBtn.addEventListener('click', () => {
            alert('Drone inspection scheduled for tomorrow at 10:00 AM.');
        });
    }
}

/**
 * Initialize sensor charts in monitoring tab
 */
function initSensorCharts() {
    if (!sensorChart || !sensorSelect || !timeRangeSelect) return;
    
    // Create dummy chart data
    const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    // Create dummy sensor data with slight variations
    const strainData = [3.2, 3.5, 3.4, 3.8, 4.0, 3.7, 3.5];
    const vibrationData = [0.5, 0.6, 0.7, 1.2, 0.9, 0.8, 0.7];
    const temperatureData = [22, 23, 25, 26, 24, 22, 21];
    
    // Create the chart
    sensorChartInstance = new Chart(sensorChart, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Strain (mm)',
                    data: strainData,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Vibration (g)',
                    data: vibrationData,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Temperature (°C)',
                    data: temperatureData,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
    
    // Handle sensor type selection
    sensorSelect.addEventListener('change', updateSensorChart);
    timeRangeSelect.addEventListener('change', updateSensorChart);
}

/**
 * Update sensor chart based on selections
 */
function updateSensorChart() {
    if (!sensorChartInstance) return;
    
    const sensorType = sensorSelect.value;
    const timeRange = timeRangeSelect.value;
    
    // Get appropriate time labels based on time range
    let labels;
    if (timeRange === 'day') {
        labels = ['12AM', '4AM', '8AM', '12PM', '4PM', '8PM', '11PM'];
    } else if (timeRange === 'week') {
        labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    } else if (timeRange === 'month') {
        labels = ['Week 1', 'Week 2', 'Week 3', 'Week 4'];
    } else {
        labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    }
    
    // Update chart data
    sensorChartInstance.data.labels = labels;
    
    // Show/hide datasets based on sensor type
    const showStrain = sensorType === 'all' || sensorType === 'strain';
    const showVibration = sensorType === 'all' || sensorType === 'vibration';
    const showTemperature = sensorType === 'all' || sensorType === 'temperature';
    
    sensorChartInstance.data.datasets[0].hidden = !showStrain;
    sensorChartInstance.data.datasets[1].hidden = !showVibration;
    sensorChartInstance.data.datasets[2].hidden = !showTemperature;
    
    // Update the chart
    sensorChartInstance.update();
}

/**
 * Initialize alert handlers
 */
function initAlerts() {
    // In a real app, this would set up handlers for alert actions
    document.querySelectorAll('.alert-list .alert').forEach(alert => {
        alert.addEventListener('click', () => {
            const title = alert.querySelector('h6').textContent;
            alert(`Alert details for: ${title}`);
        });
    });
}

/**
 * Update the "last update" time
 */
function updateLastUpdateTime() {
    const lastUpdateTime = document.getElementById('lastUpdateTime');
    if (!lastUpdateTime) return;
    
    lastUpdateTime.textContent = 'Just now';
    
    // Update every minute
    setInterval(() => {
        lastUpdateTime.textContent = 'Just now';
    }, 60000);
}

/**
 * Show loading indicator
 */
function showLoading(message = 'Loading...') {
    if (!loading || !loadingText) return;
    
    loading.classList.remove('d-none');
    loadingText.textContent = message;
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    if (!loading) return;
    
    loading.classList.add('d-none');
}

/**
 * Show error message
 */
function showError(message) {
    if (!errorMessage || !errorText) return;
    
    errorMessage.classList.remove('d-none');
    errorText.textContent = message;
    
    // Auto-hide after 5 seconds
    setTimeout(hideError, 5000);
}

/**
 * Hide error message
 */
function hideError() {
    if (!errorMessage) return;
    
    errorMessage.classList.add('d-none');
} 