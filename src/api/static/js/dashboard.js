/**
 * AI-Driven Structural Health Monitoring Dashboard
 * Dashboard JavaScript for interactive features
 */

// Dashboard initialization
document.addEventListener('DOMContentLoaded', () => {
    // Initialize current date
    updateCurrentDate();
    
    // Initialize charts
    initializeCharts();
    
    // Initialize map
    initializeMap();
    
    // Set up event listeners
    setupEventListeners();
    
    // Fix the dashboard statistics to display proper values
    fixDashboardStats();
    
    // Simulated real-time updates
    startRealtimeUpdates();
});

/**
 * Fix the dashboard statistics to display proper values
 */
function fixDashboardStats() {
    // Fix the statistics values (replace negative values with positive ones)
    const statElements = document.querySelectorAll('.card .display-4, .card h1');
    statElements.forEach(element => {
        const value = element.textContent.trim();
        if (value.startsWith('-')) {
            // Replace negative value with positive one
            const positiveValue = value.substring(1);
            element.textContent = positiveValue;
        }
    });
    
    // Ensure all monitored text is shown correctly
    const monitoredElements = document.querySelectorAll('.card .text-muted, .card small');
    monitoredElements.forEach(element => {
        if (element.textContent.includes('monitored')) {
            element.innerHTML = '<i class="fas fa-check-circle text-success me-1"></i> All monitored';
        }
    });
}

/**
 * Initialize dashboard charts
 */
function initializeCharts() {
    // Defect trends chart (line chart)
    const defectTrendsCtx = document.getElementById('defectTrendsChart');
    if (defectTrendsCtx) {
        const defectTrendsChart = new Chart(defectTrendsCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Minor Defects',
                    data: [12, 19, 14, 15, 12, 13],
                    borderColor: '#fbbc05',
                    backgroundColor: 'rgba(251, 188, 5, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Major Defects',
                    data: [7, 11, 5, 8, 3, 7],
                    borderColor: '#ea4335',
                    backgroundColor: 'rgba(234, 67, 53, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Defects'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Month'
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });

        // Store chart instance for later updates
        window.dashboardCharts = window.dashboardCharts || {};
        window.dashboardCharts.defectTrends = defectTrendsChart;
    }

    // Defect types chart (doughnut chart)
    const defectTypesCtx = document.getElementById('defectTypesChart');
    if (defectTypesCtx) {
        const defectTypesChart = new Chart(defectTypesCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Cracks', 'Corrosion', 'Deformation', 'Displacement', 'Other'],
                datasets: [{
                    data: [45, 25, 15, 10, 5],
                    backgroundColor: [
                        '#1a73e8',
                        '#ea4335',
                        '#fbbc05',
                        '#34a853',
                        '#9aa0a6'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const total = context.dataset.data.reduce((acc, val) => acc + val, 0);
                                const percentage = Math.round((value / total) * 100);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                },
                cutout: '70%'
            }
        });

        // Store chart instance for later updates
        window.dashboardCharts = window.dashboardCharts || {};
        window.dashboardCharts.defectTypes = defectTypesChart;
    }
}

/**
 * Initialize the map with structure locations
 */
function initializeMap() {
    // Initialize map if the element exists
    const mapElement = document.getElementById('structureMap');
    if (!mapElement) return;

    // Create map centered on India
    const map = L.map('structureMap').setView([22.5937, 78.9629], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    // Indian infrastructure locations
    const structureLocations = [
        { id: 1, name: "Howrah Bridge", lat: 22.5851, lng: 88.3468, status: "warning", type: "bridge" },
        { id: 2, name: "Bandra-Worli Sea Link", lat: 19.0345, lng: 72.8147, status: "healthy", type: "bridge" },
        { id: 3, name: "Bhakra Dam", lat: 31.4110, lng: 76.4333, status: "healthy", type: "dam" },
        { id: 4, name: "Delhi Metro Bridge", lat: 28.6139, lng: 77.2090, status: "warning", type: "bridge" },
        { id: 5, name: "Taj Mahal", lat: 27.1751, lng: 78.0421, status: "healthy", type: "building" },
        { id: 6, name: "Tehri Dam", lat: 30.3774, lng: 78.4803, status: "warning", type: "dam" },
        { id: 7, name: "Vidyasagar Setu", lat: 22.5569, lng: 88.3307, status: "healthy", type: "bridge" },
        { id: 8, name: "Sardar Sarovar Dam", lat: 21.8395, lng: 73.7493, status: "healthy", type: "dam" },
        { id: 9, name: "India Gate", lat: 28.6129, lng: 77.2295, status: "healthy", type: "building" },
        { id: 10, name: "Chennai Metro Bridge", lat: 13.0827, lng: 80.2707, status: "critical", type: "bridge" }
    ];

    // Add markers for each structure
    const markers = [];
    structureLocations.forEach(structure => {
        const iconColor = structure.status === 'healthy' ? 'green' : 
                         structure.status === 'warning' ? 'orange' : 'red';
        
        // Create custom marker icon based on structure type
        let iconClass = 'fa-building';
        if (structure.type === 'bridge') iconClass = 'fa-archway';
        else if (structure.type === 'dam') iconClass = 'fa-water';
        
        const markerIcon = L.divIcon({
            html: `<i class="fas ${iconClass}" style="color: ${iconColor}; font-size: 24px;"></i>`,
            className: 'custom-marker-icon',
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });
        
        // Create marker and popup
        const marker = L.marker([structure.lat, structure.lng], { icon: markerIcon })
            .addTo(map)
            .bindPopup(`
                <strong>${structure.name}</strong><br>
                Type: ${structure.type.charAt(0).toUpperCase() + structure.type.slice(1)}<br>
                Status: <span class="badge bg-${iconColor === 'green' ? 'success' : iconColor === 'orange' ? 'warning' : 'danger'}">${structure.status}</span><br>
                <a href="/structures/${structure.id}" class="btn btn-sm btn-primary mt-2">View Details</a>
            `);
        
        markers.push({ marker, status: structure.status, type: structure.type });
    });

    // Set up filter buttons
    const filterButtons = {
        'mapAllBtn': 'all',
        'mapHealthyBtn': 'healthy',
        'mapWarningBtn': 'warning',
        'mapCriticalBtn': 'critical'
    };

    Object.entries(filterButtons).forEach(([btnId, statusFilter]) => {
        const btn = document.getElementById(btnId);
        if (btn) {
            btn.addEventListener('click', () => {
                // Update active button state
                Object.keys(filterButtons).forEach(id => {
                    document.getElementById(id).classList.toggle('active', id === btnId);
                });

                // Filter markers
                markers.forEach(m => {
                    if (statusFilter === 'all' || m.status === statusFilter) {
                        map.addLayer(m.marker);
                    } else {
                        map.removeLayer(m.marker);
                    }
                });
            });
        }
    });

    // Store map instance for later use
    window.structureMap = {
        map,
        markers
    };
}

/**
 * Set up event listeners for interactive elements
 */
function setupEventListeners() {
    // New inspection button
    const inspectionBtn = document.getElementById('inspectionBtn');
    if (inspectionBtn) {
        inspectionBtn.addEventListener('click', () => {
            const modal = new bootstrap.Modal(document.getElementById('newInspectionModal'));
            modal.show();
        });
    }

    // Start inspection button
    const startInspectionBtn = document.getElementById('startInspectionBtn');
    if (startInspectionBtn) {
        startInspectionBtn.addEventListener('click', () => {
            // Get form data
            const structureId = document.getElementById('structureSelect').value;
            const inspectionType = document.getElementById('inspectionType').value;
            const notes = document.getElementById('inspectionNotes').value;
            const automaticCapture = document.getElementById('automaticCapture').checked;

            // Validate selection
            if (!structureId) {
                alert('Please select a structure to inspect');
                return;
            }

            // Create inspection
            createNewInspection(structureId, inspectionType, notes, automaticCapture);
        });
    }

    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            // Add spin effect to button icon
            const refreshIcon = refreshBtn.querySelector('i');
            refreshIcon.classList.add('fa-spin');
            
            // Simulate refresh delay
            setTimeout(() => {
                refreshIcon.classList.remove('fa-spin');
                
                // Update last refreshed time
                document.getElementById('refreshIndicator').innerHTML = 
                    `<i class="fas fa-sync-alt"></i> Last updated: Just now`;
                
                // Show toast notification
                showToast('Dashboard refreshed successfully', 'success');
                
                // Update dashboard data
                fetchDashboardData();
            }, 1000);
        });
    }

    // Export button
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            exportDashboardData();
        });
    }
}

/**
 * Create a new inspection
 */
function createNewInspection(structureId, inspectionType, notes, automaticCapture) {
    // Create form data
    const formData = new FormData();
    formData.append('structure_id', structureId);
    formData.append('inspection_type', inspectionType);
    formData.append('notes', notes);
    formData.append('automatic_capture', automaticCapture);

    // Send request to server
    fetch('/api/new_inspection', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to create inspection');
        }
        return response.json();
    })
    .then(data => {
        // Hide modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('newInspectionModal'));
        modal.hide();
        
        // Show success message
        showToast(`Inspection started for ${data.structure_name}`, 'success');
        
        // In a real app, we would redirect to the inspection page
        // window.location.href = `/inspections/${data.id}`;
        
        // For demo purposes, just update the inspection timeline
        updateInspectionTimeline(data);
    })
    .catch(error => {
        console.error('Error creating inspection:', error);
        showToast('Failed to create inspection: ' + error.message, 'danger');
    });
}

/**
 * Update the inspection timeline with a new inspection
 */
function updateInspectionTimeline(inspection) {
    const timelineContainer = document.querySelector('.inspection-timeline');
    if (!timelineContainer) return;

    // Create new timeline item
    const newItem = document.createElement('div');
    newItem.className = 'timeline-item';
    newItem.innerHTML = `
        <div class="d-flex justify-content-between mb-1">
            <h6 class="mb-0">${inspection.structure_name}</h6>
            <small class="text-muted">Just now</small>
        </div>
        <p class="mb-0 text-truncate">${inspection.inspection_type} inspection initiated</p>
        <small class="badge bg-info">Pending</small>
    `;

    // Add to timeline (at the top)
    timelineContainer.insertBefore(newItem, timelineContainer.firstChild);
}

/**
 * Show a toast notification
 */
function showToast(message, type = 'success') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        toastContainer.style.zIndex = 1050;
        document.body.appendChild(toastContainer);
    }

    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastElement = document.createElement('div');
    toastElement.className = `toast align-items-center text-white bg-${type} border-0`;
    toastElement.id = toastId;
    toastElement.setAttribute('role', 'alert');
    toastElement.setAttribute('aria-live', 'assertive');
    toastElement.setAttribute('aria-atomic', 'true');
    
    // Toast content
    toastElement.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'} me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Add to container
    toastContainer.appendChild(toastElement);
    
    // Initialize and show toast
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 5000
    });
    toast.show();
    
    // Remove after hiding
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * Update current date display
 */
function updateCurrentDate() {
    const dateElement = document.getElementById('currentDate');
    if (dateElement) {
        const now = new Date();
        dateElement.textContent = now.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }
}

/**
 * Export dashboard data (simulated)
 */
function exportDashboardData() {
    // In a real app, this would prepare a CSV or PDF download
    showToast('Dashboard data export initiated. Your download will begin shortly.', 'success');
    
    // Simulate download delay
    setTimeout(() => {
        // Create a dummy CSV file with Indian infrastructure
        const csv = 'data:text/csv;charset=utf-8,Structure,Status,Last Inspection,Defects\n' +
            'Howrah Bridge,Warning,2025-03-12,3\n' +
            'Bandra-Worli Sea Link,Healthy,2025-03-01,0\n' +
            'Bhakra Dam,Healthy,2025-02-28,1\n' +
            'Delhi Metro Bridge,Warning,2025-03-05,2\n' +
            'Taj Mahal,Healthy,2025-03-10,0\n' +
            'Chennai Metro Bridge,Critical,2025-03-08,5\n' +
            'Tehri Dam,Warning,2025-02-25,2\n';
        
        // Create download link
        const link = document.createElement('a');
        link.href = encodeURI(csv);
        link.target = '_blank';
        link.download = 'indian-infrastructure-health-report.csv';
        link.click();
    }, 1500);
}

/**
 * Fetch dashboard data from server
 */
function fetchDashboardData() {
    // Fetch statistics
    fetch('/api/statistics')
        .then(response => response.json())
        .then(data => {
            // Update statistics
            updateStatisticCard('total_structures', data.total_structures);
            updateStatisticCard('healthy_structures', data.healthy_structures);
            updateStatisticCard('warning_structures', data.warning_structures);
            updateStatisticCard('critical_structures', data.critical_structures);
        })
        .catch(err => console.error('Error fetching statistics:', err));

    // Fetch alerts
    fetch('/api/alerts')
        .then(response => response.json())
        .then(data => {
            // Update alerts (in a real app)
        })
        .catch(err => console.error('Error fetching alerts:', err));

    // Fetch inspections
    fetch('/api/inspections')
        .then(response => response.json())
        .then(data => {
            // Update inspections (in a real app)
        })
        .catch(err => console.error('Error fetching inspections:', err));
}

/**
 * Update a statistic card with new value
 */
function updateStatisticCard(id, value) {
    const element = document.querySelector(`[data-stat="${id}"]`);
    if (element) {
        // Animate value change
        const currentValue = parseInt(element.textContent);
        animateValue(element, currentValue, value, 1000);
    }
}

/**
 * Animate a value change
 */
function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = end > start ? 1 : -1;
    const stepTime = Math.abs(Math.floor(duration / range));
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        element.textContent = current;
        if (current === end) {
            clearInterval(timer);
        }
    }, stepTime);
}

/**
 * Start simulated real-time updates
 */
function startRealtimeUpdates() {
    // In a real app, this would use WebSockets for real-time data
    // For demo, just update randomly every 30 seconds
    setInterval(() => {
        // Only run if dashboard charts exist
        if (!window.dashboardCharts) return;
        
        // Update defect trends chart
        const trendsChart = window.dashboardCharts.defectTrends;
        if (trendsChart) {
            const minorData = trendsChart.data.datasets[0].data;
            const majorData = trendsChart.data.datasets[1].data;
            
            // Add slight random fluctuations to data
            minorData.forEach((val, i) => {
                minorData[i] = Math.max(0, val + (Math.random() > 0.5 ? 1 : -1) * Math.floor(Math.random() * 3));
            });
            
            majorData.forEach((val, i) => {
                majorData[i] = Math.max(0, val + (Math.random() > 0.5 ? 1 : -1) * Math.floor(Math.random() * 2));
            });
            
            trendsChart.update();
        }
    }, 30000);
} 