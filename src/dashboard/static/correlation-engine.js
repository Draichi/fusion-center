// Project Overwatch Dashboard - Main Application
let globe = null;
let refreshInterval = null;
let startTime = Date.now();
let composer = null;
let bloomPass = null;
let scene = null;
let camera = null;
let renderer = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    // Wait for all scripts to load, especially post-processing
    waitForPostProcessing(() => {
        initializeGlobe();
        loadDashboardData();
        startRefreshTimer();
        startUptimeCounter();
    });
});

// Wait for Three.js post-processing scripts to load
function waitForPostProcessing(callback, maxAttempts = 50, attempt = 0) {
    if (typeof THREE === 'undefined') {
        console.error('[BLOOM] THREE.js not loaded!');
        if (attempt < maxAttempts) {
            setTimeout(() => waitForPostProcessing(callback, maxAttempts, attempt + 1), 100);
        } else {
            console.error('[BLOOM] THREE.js failed to load after', maxAttempts, 'attempts');
            callback(); // Continue anyway
        }
        return;
    }
    
    // Check for post-processing libraries
    const hasEffectComposer = typeof THREE.EffectComposer !== 'undefined';
    const hasRenderPass = typeof THREE.RenderPass !== 'undefined';
    const hasUnrealBloomPass = typeof THREE.UnrealBloomPass !== 'undefined';
    
    if (hasEffectComposer && hasRenderPass && hasUnrealBloomPass) {
        console.log('[BLOOM] ✅ All post-processing libraries loaded');
        callback();
    } else {
        if (attempt < maxAttempts) {
            if (attempt % 10 === 0) {
                console.log(`[BLOOM] Waiting for post-processing libraries... (attempt ${attempt}/${maxAttempts})`);
                console.log('[BLOOM] EffectComposer:', hasEffectComposer);
                console.log('[BLOOM] RenderPass:', hasRenderPass);
                console.log('[BLOOM] UnrealBloomPass:', hasUnrealBloomPass);
            }
            setTimeout(() => waitForPostProcessing(callback, maxAttempts, attempt + 1), 100);
        } else {
            console.warn('[BLOOM] ⚠️ Post-processing libraries not fully loaded, continuing without bloom');
            console.warn('[BLOOM] EffectComposer:', hasEffectComposer);
            console.warn('[BLOOM] RenderPass:', hasRenderPass);
            console.warn('[BLOOM] UnrealBloomPass:', hasUnrealBloomPass);
            callback(); // Continue anyway, will use material-based glow
        }
    }
}

// Initialize 3D Globe
function initializeGlobe() {
    const container = document.getElementById('globe-container');
    
    if (!container) {
        console.error('Globe container not found');
        return;
    }
    
    // Clear container first
    container.innerHTML = '';
    
    // Ensure container has proper dimensions
    const containerRect = container.getBoundingClientRect();
    if (containerRect.width === 0 || containerRect.height === 0) {
        // Wait a bit for layout to settle
        setTimeout(initializeGlobe, 100);
        return;
    }
    
    try {
        // Create globe instance
        globe = Globe()
            .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
            .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
            .showAtmosphere(true)
            .atmosphereColor('#00ff41')
            .atmosphereAltitude(0.15)
            .pointOfView({ lat: 0, lng: 0, altitude: 2.5 })
            .pointColor(() => '#00ff41')
            .pointRadius(0.5)
            .pointAltitude(0.01)
            .pointLabel(() => '')
            .pointsData([])
            .pointResolution(16)
            .pointsMerge(false)
            .width(containerRect.width)
            .height(containerRect.height)
            (container);
        
        // Wait for globe to render and apply neon effects
        setTimeout(() => {
            const globeEl = container.querySelector('canvas');
            if (globeEl) {
                globeEl.style.filter = 'drop-shadow(0 0 20px rgba(0, 255, 65, 0.5))';
            }
            
            // Setup neon effect with post-processing (only if libraries are loaded)
            if (typeof THREE.EffectComposer !== 'undefined' && typeof THREE.UnrealBloomPass !== 'undefined') {
                const bloomSetup = setupNeonEffect();
                
                // Verify bloom is working after a delay
                setTimeout(() => {
                    verifyBloomSetup();
                }, 1000);
            } else {
                console.warn('[BLOOM] Post-processing not available, using material-based glow only');
            }
        }, 500);
        
        // Handle window resize
        let resizeTimeout;
        const handleResize = () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (globe && container) {
                    const rect = container.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        globe.width(rect.width);
                        globe.height(rect.height);
                    }
                }
            }, 250);
        };
        
        window.addEventListener('resize', handleResize);
    } catch (error) {
        console.error('Error initializing globe:', error);
        container.innerHTML = '<div class="error">Failed to initialize globe. Please check browser console.</div>';
    }
}

// Load all dashboard data
async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard');
        const data = await response.json();
        
        if (data.status === 'success') {
            updateNews(data.news);
            updateThermalAnomalies(data.thermal_anomalies);
            updateTelegram(data.telegram);
            updateThreatIntel(data.threat_intel);
            updateLastUpdate(data.timestamp);
        } else {
            showError('Failed to load dashboard data');
        }
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showError('Error connecting to server');
    }
}

// Update News Section
function updateNews(newsData) {
    const content = document.getElementById('news-content');
    const status = document.getElementById('news-status');
    
    if (newsData.status === 'error') {
        status.textContent = 'ERROR';
        status.className = 'panel-status error';
        content.innerHTML = `<div class="error">${newsData.error_message || 'Failed to load news'}</div>`;
        return;
    }
    
    status.textContent = `OK (${newsData.count})`;
    status.className = 'panel-status success';
    
    if (newsData.count === 0) {
        content.innerHTML = '<div class="empty-state">No news articles found</div>';
        return;
    }
    
    const articles = newsData.articles || [];
    content.innerHTML = articles.map(article => `
        <div class="news-item">
            <div class="news-title">${escapeHtml(article.title)}</div>
            <div class="news-meta">
                <span class="news-domain">${escapeHtml(article.domain)}</span>
                <span> | </span>
                <span>${formatDate(article.seendate)}</span>
            </div>
            <a href="${article.url}" target="_blank" class="news-link">[READ MORE]</a>
        </div>
    `).join('');
}

// Update Thermal Anomalies Section
function updateThermalAnomalies(anomaliesData) {
    const status = document.getElementById('anomalies-status');
    const countEl = document.getElementById('anomalies-count');
    
    if (anomaliesData.status === 'error') {
        status.textContent = 'ERROR';
        status.className = 'panel-status error';
        countEl.textContent = '0';
        return;
    }
    
    const anomalies = anomaliesData.anomalies || [];
    const count = anomalies.length;
    
    status.textContent = `OK (${count})`;
    status.className = 'panel-status success';
    countEl.textContent = count;
    
    // Update globe with anomaly points
    if (globe && count > 0) {
        try {
            // Create points with neon glow effect using multiple layers
            const points = [];
            
            anomalies.forEach(anomaly => {
                const brightness = parseFloat(anomaly.brightness) || 300;
                const confidence = (anomaly.confidence || '').toLowerCase();
                const lat = parseFloat(anomaly.latitude) || 0;
                const lng = parseFloat(anomaly.longitude) || 0;
                
                // Calculate size based on brightness - MUCH smaller (0.3 to 1.0)
                const normalizedBrightness = Math.max(300, Math.min(500, brightness));
                const baseSize = 0.3 + ((normalizedBrightness - 300) / 200) * 0.7;
                
                // Calculate color - use neon colors matching dashboard theme
                const color = getAnomalyColor(brightness, confidence);
                
                // Create label
                const date = anomaly.acq_date || '';
                const time = anomaly.acq_time || '';
                const zone = anomaly.zone_name || '';
                const label = `${zone ? zone + ' - ' : ''}${date} ${time} (${Math.round(brightness)}K)`;
                
                // Create multiple glow layers for neon effect
                // Outer glow (largest, most transparent)
                points.push({
                    lat: lat,
                    lng: lng,
                    size: baseSize * 4.0,
                    color: color,
                    brightness: brightness,
                    label: '',
                    layer: 'outer-glow',
                });
                
                // Middle glow (medium size, semi-transparent)
                points.push({
                    lat: lat,
                    lng: lng,
                    size: baseSize * 2.5,
                    color: color,
                    brightness: brightness,
                    label: '',
                    layer: 'middle-glow',
                });
                
                // Inner glow (smaller, more opaque)
                points.push({
                    lat: lat,
                    lng: lng,
                    size: baseSize * 1.5,
                    color: color,
                    brightness: brightness,
                    label: '',
                    layer: 'inner-glow',
                });
                
                // Core point (smallest, fully opaque, brightest)
                points.push({
                    lat: lat,
                    lng: lng,
                    size: baseSize,
                    color: color,
                    brightness: brightness,
                    confidence: confidence,
                    label: label,
                    layer: 'core',
                });
            });
            
            globe.pointsData(points);
            
            // Configure point rendering with neon glow properties
            globe.pointColor(d => d.color || '#00ff41');
            globe.pointRadius(d => {
                // Different sizes for different layers
                return d.size || 0.3;
            });
            globe.pointAltitude(d => {
                // All points at same altitude, but vary slightly for depth
                const baseAlt = 0.01;
                const layerOffset = {
                    'outer-glow': -0.001,
                    'middle-glow': 0,
                    'inner-glow': 0.001,
                    'core': 0.002,
                };
                return baseAlt + (layerOffset[d.layer] || 0);
            });
            globe.pointLabel(d => d.label || '');
            globe.pointResolution(16); // Higher resolution for smoother points
            
            // Apply neon colors - core points get full color, glow layers get brighter colors
            const opacityMap = {
                'outer-glow': 0.15,
                'middle-glow': 0.35,
                'inner-glow': 0.65,
                'core': 1.0,
            };
            
            globe.pointColor(d => {
                const baseColor = d.color || '#00ff41';
                const opacity = opacityMap[d.layer] || 1.0;
                
                // For glow layers, use brighter colors
                if (d.layer !== 'core') {
                    return adjustColorBrightness(baseColor, 1.0 + (1.0 - opacity) * 0.5);
                }
                return baseColor;
            });
            
            // Apply neon material to points after they're rendered
            // Wait longer since MCP can take ~30s to respond with data
            // Use retry mechanism since Globe.gl may not have rendered points yet
            let materialAttempts = 0;
            const maxMaterialAttempts = 20; // Try for up to 10 seconds (20 * 500ms)
            
            const applyMaterialWithRetry = () => {
                materialAttempts++;
                const applied = applyNeonMaterialToPoints();
                
                if (!applied && materialAttempts < maxMaterialAttempts) {
                    // Retry if material wasn't applied (scene might not be ready)
                    setTimeout(applyMaterialWithRetry, 500);
                } else if (applied) {
                    // Material applied successfully, try to setup bloom
                    if (!composer && typeof THREE.EffectComposer !== 'undefined') {
                        console.log('[BLOOM] Re-attempting bloom setup after material application');
                        setupNeonEffect();
                        
                        // Verify bloom after delay
                        setTimeout(() => {
                            verifyBloomSetup();
                        }, 1000);
                    }
                } else if (materialAttempts >= maxMaterialAttempts) {
                    console.warn('[MATERIAL] Max attempts reached, material may not be applied');
                }
            };
            
            // Start with initial delay, then retry
            setTimeout(applyMaterialWithRetry, 1000);
            
            // Apply neon material to points after they're rendered
            setTimeout(() => {
                applyNeonMaterialToPoints();
                
                // Re-setup bloom if needed (points might have been recreated)
                if (!composer) {
                    console.log('[BLOOM] Re-attempting bloom setup after points update');
                    setupNeonEffect();
                }
            }, 200);
        } catch (error) {
            console.error('Error updating globe points:', error);
        }
    } else if (globe) {
        try {
            globe.pointsData([]);
        } catch (error) {
            console.error('Error clearing globe points:', error);
        }
    }
}

// Get color based on brightness (temperature) and confidence
// Uses neon colors matching dashboard theme
function getAnomalyColor(brightness, confidence) {
    // Normalize brightness to 0-1 range (assuming 300-500K range)
    const normalizedBrightness = Math.max(0, Math.min(1, (brightness - 300) / 200));
    
    // High confidence or high brightness = neon red/orange
    const conf = (confidence || '').toLowerCase();
    const isHighConfidence = conf.includes('high') || conf.includes('nominal');
    
    if (isHighConfidence || normalizedBrightness > 0.7) {
        // Neon red for high confidence/hot anomalies
        return '#ff0044';
    } else if (normalizedBrightness > 0.5) {
        // Neon orange-red for medium-high temperature
        return '#ff4400';
    } else if (normalizedBrightness > 0.3) {
        // Neon orange for medium temperature
        return '#ff6600';
    } else {
        // Neon yellow-orange for lower temperature (closer to dashboard green)
        return '#ffaa00';
    }
}

// Helper function to adjust color brightness (for glow effect simulation)
function adjustColorBrightness(hex, factor) {
    // Remove # if present
    hex = hex.replace('#', '');
    
    // Parse RGB
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    
    // Adjust brightness (factor > 1 makes brighter)
    const newR = Math.min(255, Math.floor(r * factor));
    const newG = Math.min(255, Math.floor(g * factor));
    const newB = Math.min(255, Math.floor(b * factor));
    
    // Convert back to hex
    return '#' + 
        newR.toString(16).padStart(2, '0') + 
        newG.toString(16).padStart(2, '0') + 
        newB.toString(16).padStart(2, '0');
}

// Setup neon effect using Three.js post-processing (Bloom)
function setupNeonEffect() {
    if (!globe || !window.THREE) {
        console.warn('[BLOOM] Globe or THREE.js not available');
        return false;
    }
    
    try {
        // Check if post-processing libraries are loaded
        if (typeof THREE.EffectComposer === 'undefined') {
            console.error('[BLOOM] Three.js post-processing not loaded. Check script tags.');
            return false;
        }
        
        console.log('[BLOOM] Post-processing libraries loaded');
        
        // Access Three.js objects from Globe.gl
        renderer = globe.renderer();
        camera = globe.camera();
        
        if (!renderer || !camera) {
            console.warn('[BLOOM] Renderer or camera not available');
            return false;
        }
        
        console.log('[BLOOM] Renderer and camera accessed:', { renderer, camera });
        
        // Access scene from Globe.gl - try multiple methods
        scene = null;
        
        // Method 1: Direct access if Globe.gl exposes it
        if (globe.scene && typeof globe.scene === 'function') {
            scene = globe.scene();
        }
        
        // Method 2: Access via renderer's internal structure
        if (!scene && renderer.domElement) {
            // Globe.gl might store scene reference
            const container = renderer.domElement.parentElement;
            if (container && container.__globeScene) {
                scene = container.__globeScene;
            }
        }
        
        // Method 3: Access via Globe.gl's internal renderer
        if (!scene && renderer._globeRenderer) {
            scene = renderer._globeRenderer.scene;
        }
        
        // Method 4: Find scene by traversing renderer's children
        if (!scene) {
            // Try to find scene in renderer's internal structure
            for (const key in renderer) {
                if (renderer[key] && renderer[key].isScene) {
                    scene = renderer[key];
                    break;
                }
            }
        }
        
        // Method 5: Access via Globe.gl's scene() method if available
        if (!scene && typeof globe.scene === 'function') {
            try {
                scene = globe.scene();
            } catch (e) {
                console.warn('[BLOOM] Could not get scene via globe.scene()');
            }
        }
        
        if (!scene) {
            console.warn('[BLOOM] Could not access scene. Trying alternative approach...');
            // Fallback: we'll hook into the render loop differently
            return setupBloomAlternative();
        }
        
        console.log('[BLOOM] Scene accessed:', scene);
        
        // Create effect composer for post-processing
        const pixelRatio = window.devicePixelRatio || 1;
        const width = renderer.domElement.width || renderer.domElement.clientWidth;
        const height = renderer.domElement.height || renderer.domElement.clientHeight;
        
        composer = new THREE.EffectComposer(renderer);
        console.log('[BLOOM] EffectComposer created');
        
        // Create render pass
        const renderPass = new THREE.RenderPass(scene, camera);
        composer.addPass(renderPass);
        console.log('[BLOOM] RenderPass added');
        
        // Create bloom pass for neon glow effect
        bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(width * pixelRatio, height * pixelRatio),
            2.5,  // strength - intensity of the bloom (higher = more glow)
            0.6,  // radius - size of the glow (0-1)
            0.85  // threshold - only pixels brighter than this will glow
        );
        composer.addPass(bloomPass);
        console.log('[BLOOM] UnrealBloomPass added with strength:', bloomPass.strength);
        
        // Hook into Globe.gl's render loop
        const originalRender = renderer.render.bind(renderer);
        
        // Store original for potential restoration
        renderer._originalRender = originalRender;
        
        // Override renderer's render method to use composer
        renderer.render = function(sceneToRender, cameraToRender) {
            if (composer && sceneToRender && cameraToRender) {
                // Update composer size if needed
                const w = renderer.domElement.width || renderer.domElement.clientWidth;
                const h = renderer.domElement.height || renderer.domElement.clientHeight;
                
                if (composer.setSize) {
                    composer.setSize(w, h);
                }
                
                // Render with bloom effect
                composer.render();
                return; // Don't call original render
            }
            
            // Fallback to original render if composer not ready
            originalRender(sceneToRender, cameraToRender);
        };
        
        console.log('[BLOOM] ✅ Render method overridden to use composer');
        
        console.log('[BLOOM] ✅ Bloom effect setup complete!');
        console.log('[BLOOM] Composer:', composer);
        console.log('[BLOOM] Bloom pass:', bloomPass);
        
        return true;
    } catch (error) {
        console.error('[BLOOM] Error setting up bloom:', error);
        console.error('[BLOOM] Stack:', error.stack);
        return false;
    }
}

// Alternative bloom setup if direct scene access fails
function setupBloomAlternative() {
    console.log('[BLOOM] Trying alternative bloom setup...');
    
    // Use a render target approach or CSS filters as fallback
    // For now, we'll rely on material emission which still creates a glow effect
    console.log('[BLOOM] Using material-based neon effect (emission)');
    return false;
}

// Verify bloom setup is working
function verifyBloomSetup() {
    console.log('[BLOOM] Verifying bloom setup...');
    console.log('[BLOOM] Composer exists:', !!composer);
    console.log('[BLOOM] Bloom pass exists:', !!bloomPass);
    console.log('[BLOOM] Scene exists:', !!scene);
    console.log('[BLOOM] Camera exists:', !!camera);
    console.log('[BLOOM] Renderer exists:', !!renderer);
    
    if (composer && bloomPass) {
        console.log('[BLOOM] ✅ Bloom is configured!');
        console.log('[BLOOM] Bloom strength:', bloomPass.strength);
        console.log('[BLOOM] Bloom radius:', bloomPass.radius);
        console.log('[BLOOM] Bloom threshold:', bloomPass.threshold);
        
        // Test if composer can render
        if (scene && camera && renderer) {
            try {
                composer.render();
                console.log('[BLOOM] ✅ Composer render test successful!');
            } catch (e) {
                console.error('[BLOOM] ❌ Composer render test failed:', e);
            }
        }
    } else {
        console.warn('[BLOOM] ⚠️ Bloom not fully configured. Check logs above.');
    }
}

// Apply neon material to points with emission for real neon effect
function applyNeonMaterialToPoints() {
    if (!window.THREE) {
        console.warn('[MATERIAL] THREE.js not available');
        return false;
    }
    
    if (!globe) {
        console.warn('[MATERIAL] Globe not initialized');
        return false;
    }
    
    // Try to get scene if not already set
    if (!scene) {
        renderer = globe.renderer();
        if (renderer) {
            // Try multiple methods to get scene
            if (globe.scene && typeof globe.scene === 'function') {
                try {
                    scene = globe.scene();
                } catch (e) {
                    console.warn('[MATERIAL] Could not get scene via globe.scene()');
                }
            }
        }
    }
    
    // If still no scene, try to access via renderer
    if (!scene && renderer) {
        // Globe.gl might store scene in renderer's internal structure
        // Try accessing via renderer's scene property or parent
        if (renderer.domElement && renderer.domElement.parentElement) {
            const container = renderer.domElement.parentElement;
            // Check if scene is stored somewhere in the container
            if (container.__globeScene) {
                scene = container.__globeScene;
            }
        }
    }
    
    if (!scene) {
        // Last resort: try to find scene by checking renderer's internal properties
        if (renderer) {
            for (const key in renderer) {
                if (renderer[key] && renderer[key].isScene) {
                    scene = renderer[key];
                    break;
                }
            }
        }
    }
    
    if (!scene) {
        console.warn('[MATERIAL] Could not access scene for material application');
        return false;
    }
    
    try {
        let pointCount = 0;
        
        scene.traverse((object) => {
            // Find point meshes created by Globe.gl
            if (object.isMesh && object.material && object.geometry) {
                // Check if this looks like a point mesh
                const isPointMesh = object.geometry.type === 'BufferGeometry' &&
                                   (object.material.type === 'MeshBasicMaterial' || 
                                    object.material.type === 'PointsMaterial');
                
                if (isPointMesh && !object.userData.neonMaterialApplied) {
                    pointCount++;
                    
                    // Get color from current material or use default neon green
                    let currentColor = new THREE.Color('#00ff41');
                    if (object.material.color) {
                        currentColor = object.material.color.clone();
                    }
                    
                    // Create neon material with strong emission
                    // High emissiveIntensity creates glow even without bloom
                    object.material = new THREE.MeshBasicMaterial({
                        color: currentColor,
                        transparent: true,
                        opacity: 0.95,
                        emissive: currentColor,
                        emissiveIntensity: 5.0, // Very strong emission for visible glow
                    });
                    
                    // Mark as neon point
                    object.userData.isNeonPoint = true;
                    object.userData.neonMaterialApplied = true;
                }
            }
        });
        
        if (pointCount > 0) {
            console.log(`[MATERIAL] ✅ Applied neon material to ${pointCount} point meshes`);
            console.log(`[MATERIAL] Material uses emissiveIntensity: 5.0 for glow effect`);
            return true;
        } else {
            console.log('[MATERIAL] No point meshes found yet (Globe.gl may still be rendering)');
            return false;
        }
    } catch (error) {
        console.error('[MATERIAL] Error applying neon material:', error);
        return false;
    }
}

// Update Telegram Section
function updateTelegram(telegramData) {
    const content = document.getElementById('telegram-content');
    const status = document.getElementById('telegram-status');
    
    if (telegramData.status === 'error') {
        status.textContent = 'ERROR';
        status.className = 'panel-status error';
        content.innerHTML = `<div class="error">${telegramData.error_message || 'Telegram not configured'}</div>`;
        return;
    }
    
    status.textContent = `OK (${telegramData.count})`;
    status.className = 'panel-status success';
    
    if (telegramData.count === 0) {
        content.innerHTML = '<div class="empty-state">No Telegram messages found</div>';
        return;
    }
    
    const messages = telegramData.messages || [];
    content.innerHTML = messages.map(msg => `
        <div class="telegram-item">
            <div class="telegram-channel">@${escapeHtml(msg.channel_username)}</div>
            <div class="telegram-text">${escapeHtml(truncateText(msg.text, 200))}</div>
            <div class="telegram-meta">
                ${formatDate(msg.date)} | Views: ${msg.views || 'N/A'}
                ${msg.url ? ` | <a href="${msg.url}" target="_blank" class="news-link">[LINK]</a>` : ''}
            </div>
        </div>
    `).join('');
}

// Update Threat Intel Section
function updateThreatIntel(threatData) {
    const content = document.getElementById('threat-content');
    const status = document.getElementById('threat-status');
    
    if (threatData.status === 'error') {
        status.textContent = 'ERROR';
        status.className = 'panel-status error';
        content.innerHTML = `<div class="error">${threatData.error_message || 'OTX not configured'}</div>`;
        return;
    }
    
    status.textContent = `OK (${threatData.count})`;
    status.className = 'panel-status success';
    
    if (threatData.count === 0) {
        content.innerHTML = '<div class="empty-state">No threat pulses found</div>';
        return;
    }
    
    const pulses = threatData.pulses || [];
    content.innerHTML = pulses.map(pulse => `
        <div class="threat-item">
            <div class="threat-name">${escapeHtml(pulse.name)}</div>
            <div class="threat-description">${escapeHtml(truncateText(pulse.description || 'No description', 150))}</div>
            <div class="threat-meta">
                Indicators: ${pulse.indicator_count || 0} | 
                Author: ${escapeHtml(pulse.author_name || 'Unknown')}
            </div>
            ${pulse.tags && pulse.tags.length > 0 ? `
                <div class="threat-tags">
                    ${pulse.tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}
                </div>
            ` : ''}
        </div>
    `).join('');
}

// Update last update timestamp
function updateLastUpdate(timestamp) {
    const el = document.getElementById('last-update');
    if (el && timestamp) {
        const date = new Date(timestamp);
        el.textContent = date.toLocaleTimeString();
    }
}

// Start refresh timer
function startRefreshTimer() {
    let seconds = 60;
    const timerEl = document.getElementById('refresh-timer');
    
    refreshInterval = setInterval(() => {
        seconds--;
        if (timerEl) {
            timerEl.textContent = `${seconds}s`;
        }
        
        if (seconds <= 0) {
            seconds = 60;
            loadDashboardData();
        }
    }, 1000);
}

// Start uptime counter
function startUptimeCounter() {
    setInterval(() => {
        const uptimeEl = document.getElementById('uptime');
        if (uptimeEl) {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(elapsed / 3600);
            const minutes = Math.floor((elapsed % 3600) / 60);
            const seconds = elapsed % 60;
            uptimeEl.textContent = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }
    }, 1000);
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch (e) {
        return dateString;
    }
}

function showError(message) {
    console.error('Dashboard error:', message);
    // Could show a toast notification here
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    if (globe) {
        globe._destructor();
    }
});

