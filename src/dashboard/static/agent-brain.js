/**
 * Agent Brain Dashboard - WebSocket client and UI controller
 *
 * Handles real-time visualization of agent_v2 research process.
 */

class AgentBrainDashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.toolCallCount = 0;
        this.globe = null;

        // Current state
        this.state = {
            isLive: false,
            sessionId: null,
            currentPhase: 'idle',
            data: null,
        };

        // Phase order for progress tracking
        this.phaseOrder = [
            'planning',
            'decomposition',
            'hypothesis',
            'gathering',
            'analysis',
            'reflection',
            'verification',
            'synthesis',
            'complete'
        ];

        this.init();
    }

    init() {
        this.initWebSocket();
        this.initEventListeners();
        this.initGlobe();
    }

    // =========================================================================
    // WebSocket Management
    // =========================================================================

    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/agent`;

        console.log('[WS] Connecting to:', wsUrl);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('[WS] Connected');
            this.reconnectAttempts = 0;
            this.updateConnectionStatus(true);
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (e) {
                console.error('[WS] Error parsing message:', e);
            }
        };

        this.ws.onclose = () => {
            console.log('[WS] Disconnected');
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('[WS] Error:', error);
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('[WS] Max reconnect attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => this.initWebSocket(), delay);
    }

    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        const wsStatusEl = document.getElementById('ws-status');
        const dot = statusEl?.querySelector('.status-dot');
        const text = statusEl?.querySelector('.status-text');

        if (connected) {
            dot?.classList.remove('disconnected');
            dot?.classList.add('connected');
            if (text) text.textContent = 'CONNECTED';
            if (wsStatusEl) wsStatusEl.textContent = 'CONNECTED';
        } else {
            dot?.classList.remove('connected');
            dot?.classList.add('disconnected');
            if (text) text.textContent = 'DISCONNECTED';
            if (wsStatusEl) wsStatusEl.textContent = 'DISCONNECTED';
        }
    }

    // =========================================================================
    // Message Handling
    // =========================================================================

    handleMessage(message) {
        console.log('[WS] Message:', message.type, message);

        switch (message.type) {
            case 'connection_status':
                this.handleConnectionStatus(message);
                break;

            case 'historical_session':
                this.handleHistoricalSession(message);
                break;

            case 'session_start':
                this.handleSessionStart(message);
                break;

            case 'phase_change':
                this.handlePhaseChange(message);
                break;

            case 'tool_call':
                this.handleToolCall(message);
                break;

            case 'session_complete':
                this.handleSessionComplete(message);
                break;

            case 'error':
                this.handleError(message);
                break;

            case 'pong':
                // Keep-alive response
                break;

            default:
                console.warn('[WS] Unknown message type:', message.type);
        }
    }

    handleConnectionStatus(message) {
        this.state.isLive = message.is_live;
        this.state.sessionId = message.session_id;
        this.state.currentPhase = message.current_phase || 'idle';
        this.updateLiveIndicator();

        if (message.session_id) {
            this.updateSessionId(message.session_id);
        }
    }

    handleHistoricalSession(message) {
        this.state.isLive = false;
        this.state.data = message.data;
        this.updateLiveIndicator();

        if (message.data) {
            this.updateSessionId(message.data.id || '--');
            if (message.data.state) {
                this.updateAllPanels(message.data.state);
                this.updatePhaseDisplay(message.data.current_phase || 'complete');
            }
        }
    }

    handleSessionStart(message) {
        this.state.isLive = true;
        this.state.sessionId = message.session_id;
        this.state.currentPhase = 'planning';
        this.toolCallCount = 0;

        this.updateLiveIndicator();
        this.updateSessionId(message.session_id);
        this.resetPanels();
        this.updatePhaseDisplay('planning');
        this.updateSystemStatus('RESEARCH STARTED');

        // Update iteration display
        if (message.data) {
            document.getElementById('iteration-max').textContent = message.data.max_iterations || 5;
        }
    }

    handlePhaseChange(message) {
        this.state.isLive = true;
        this.state.currentPhase = message.phase;
        this.state.data = message.data;

        this.updatePhaseDisplay(message.phase);
        this.updateAllPanels(message.data);
        this.updateSystemStatus(`PHASE: ${message.phase.toUpperCase()}`);
    }

    handleToolCall(message) {
        this.toolCallCount++;
        this.addToolCall(message);
    }

    handleSessionComplete(message) {
        this.state.isLive = false;
        this.state.currentPhase = 'complete';
        this.state.data = message.data;

        this.updatePhaseDisplay('complete');
        this.updateLiveIndicator();
        this.updateSystemStatus('RESEARCH COMPLETE');

        if (message.data) {
            this.updateAllPanels(message.data);
        }
    }

    handleError(message) {
        console.error('[WS] Error from server:', message.error);
        this.updateSystemStatus(`ERROR: ${message.error}`);

        const statusEl = document.getElementById('input-status');
        if (statusEl) {
            statusEl.textContent = `Error: ${message.error}`;
            statusEl.className = 'input-status error';
        }
    }

    // =========================================================================
    // UI Updates
    // =========================================================================

    updateLiveIndicator() {
        const indicator = document.getElementById('live-indicator');
        const badge = indicator?.querySelector('.live-badge');

        if (badge) {
            if (this.state.isLive) {
                badge.textContent = 'LIVE';
                badge.classList.remove('historical');
                badge.classList.add('live');
            } else {
                badge.textContent = 'HISTORICAL';
                badge.classList.remove('live');
                badge.classList.add('historical');
            }
        }
    }

    updateSessionId(sessionId) {
        const el = document.getElementById('session-id');
        if (el) {
            el.textContent = sessionId ? sessionId.substring(0, 8) + '...' : '--';
            el.title = sessionId || '';
        }
    }

    updateSystemStatus(status) {
        const el = document.getElementById('system-status');
        if (el) {
            el.textContent = status;
        }
    }

    updatePhaseDisplay(phase) {
        // Normalize phase name
        const normalizedPhase = phase.replace('_complete', '').replace('_', '');

        // Update phase items
        document.querySelectorAll('.phase-item').forEach(item => {
            const itemPhase = item.dataset.phase;
            const phaseIndex = this.phaseOrder.indexOf(normalizedPhase);
            const itemIndex = this.phaseOrder.indexOf(itemPhase);

            item.classList.remove('active', 'completed', 'pending');

            if (itemIndex < phaseIndex || phase === 'complete') {
                item.classList.add('completed');
            } else if (itemIndex === phaseIndex) {
                item.classList.add('active');
            } else {
                item.classList.add('pending');
            }
        });

        // Update connectors
        document.querySelectorAll('.phase-connector').forEach((connector, index) => {
            const phaseIndex = this.phaseOrder.indexOf(normalizedPhase);

            if (index < phaseIndex || phase === 'complete') {
                connector.classList.add('completed');
            } else {
                connector.classList.remove('completed');
            }
        });
    }

    updateAllPanels(data) {
        if (!data) return;

        // Update iteration
        this.updateIteration(data.iteration, data.max_iterations);

        // Update findings count
        this.updateFindingsCount(data.findings_count || 0);

        // Update objectives
        this.updateObjectives(data.research_plan);

        // Update sub-tasks
        this.updateSubTasks(data.sub_tasks);

        // Update hypotheses
        this.updateHypotheses(data.hypotheses);

        // Update key insights
        this.updateInsights(data.key_insights);

        // Update correlations
        this.updateCorrelations(data.correlations);

        // Update reflection notes
        this.updateReflections(data.reflection_notes);

        // Update uncertainties
        this.updateUncertainties(data.uncertainties);

        // Update globe with regions
        this.updateGlobeWithRegions(data.research_plan);
    }

    updateIteration(current, max) {
        document.getElementById('iteration-current').textContent = current || 0;
        document.getElementById('iteration-max').textContent = max || 5;

        const fill = document.getElementById('iteration-fill');
        if (fill && max > 0) {
            const percentage = ((current || 0) / max) * 100;
            fill.style.width = `${percentage}%`;
        }
    }

    updateFindingsCount(count) {
        const el = document.getElementById('findings-count');
        if (el) el.textContent = count;
    }

    updateObjectives(researchPlan) {
        const content = document.getElementById('objectives-content');
        const status = document.getElementById('objectives-status');

        if (!researchPlan || !researchPlan.objectives) {
            content.innerHTML = '<div class="empty-state">No objectives yet...</div>';
            status.textContent = '--';
            return;
        }

        const objectives = researchPlan.objectives;
        status.textContent = objectives.length;

        content.innerHTML = objectives.map((obj, i) => `
            <div class="objective-item">
                <span class="objective-number">${i + 1}.</span>
                <span class="objective-text">${this.escapeHtml(obj)}</span>
            </div>
        `).join('');
    }

    updateSubTasks(subTasks) {
        const content = document.getElementById('subtasks-content');
        const status = document.getElementById('subtasks-status');

        if (!subTasks || subTasks.length === 0) {
            content.innerHTML = '<div class="empty-state">No sub-tasks yet...</div>';
            status.textContent = '0';
            return;
        }

        status.textContent = subTasks.length;

        content.innerHTML = subTasks.map(task => `
            <div class="subtask-item ${task.status || 'pending'}">
                <span class="subtask-id">[${task.id}]</span>
                <span class="subtask-desc">${this.escapeHtml(task.description)}</span>
                <span class="subtask-focus">(${task.focus_area || 'thematic'})</span>
            </div>
        `).join('');
    }

    updateHypotheses(hypotheses) {
        const content = document.getElementById('hypotheses-content');
        const status = document.getElementById('hypotheses-status');

        if (!hypotheses || hypotheses.length === 0) {
            content.innerHTML = '<div class="empty-state">No hypotheses yet...</div>';
            status.textContent = '0';
            return;
        }

        status.textContent = hypotheses.length;

        content.innerHTML = hypotheses.map(h => {
            const confidence = (h.confidence || 0) * 100;
            const statusClass = h.status || 'proposed';

            return `
                <div class="hypothesis-item ${statusClass}">
                    <div class="hypothesis-header">
                        <span class="hypothesis-id">[${h.id}]</span>
                        <span class="hypothesis-status">${statusClass.toUpperCase()}</span>
                    </div>
                    <div class="hypothesis-statement">${this.escapeHtml(h.statement)}</div>
                    <div class="hypothesis-confidence">
                        <span class="confidence-label">Confidence:</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                        <span class="confidence-value">${Math.round(confidence)}%</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    updateInsights(insights) {
        const content = document.getElementById('insights-content');
        const status = document.getElementById('insights-status');

        if (!insights || insights.length === 0) {
            content.innerHTML = '<div class="empty-state">No insights yet...</div>';
            status.textContent = '0';
            return;
        }

        status.textContent = insights.length;

        content.innerHTML = insights.map((insight, i) => `
            <div class="insight-item">
                <span class="insight-number">${i + 1}.</span>
                <span class="insight-text">${this.escapeHtml(insight)}</span>
            </div>
        `).join('');
    }

    updateCorrelations(correlations) {
        const content = document.getElementById('correlations-content');
        const status = document.getElementById('correlations-status');

        if (!correlations || correlations.length === 0) {
            content.innerHTML = '<div class="empty-state">No correlations found...</div>';
            status.textContent = '0';
            return;
        }

        status.textContent = correlations.length;

        content.innerHTML = correlations.map(corr => `
            <div class="correlation-item">
                <span class="correlation-icon">&#x2194;</span>
                <span class="correlation-text">${this.escapeHtml(corr)}</span>
            </div>
        `).join('');
    }

    updateReflections(reflectionNotes) {
        const content = document.getElementById('reflections-content');
        const status = document.getElementById('reflections-status');

        if (!reflectionNotes || reflectionNotes.length === 0) {
            content.innerHTML = '<div class="empty-state">No reflections yet...</div>';
            status.textContent = '0';
            return;
        }

        status.textContent = reflectionNotes.length;

        content.innerHTML = reflectionNotes.map(note => `
            <div class="reflection-item ${note.severity || 'info'}">
                <div class="reflection-header">
                    <span class="reflection-category">[${(note.category || 'note').toUpperCase()}]</span>
                    <span class="reflection-severity ${note.severity || 'info'}">${(note.severity || 'info').toUpperCase()}</span>
                </div>
                <div class="reflection-content">${this.escapeHtml(note.content)}</div>
            </div>
        `).join('');
    }

    updateUncertainties(uncertainties) {
        const content = document.getElementById('uncertainties-content');
        const status = document.getElementById('uncertainties-status');

        if (!uncertainties || uncertainties.length === 0) {
            content.innerHTML = '<div class="empty-state">No uncertainties identified...</div>';
            status.textContent = '0';
            return;
        }

        status.textContent = uncertainties.length;

        content.innerHTML = uncertainties.map(unc => `
            <div class="uncertainty-item">
                <span class="uncertainty-icon">?</span>
                <span class="uncertainty-text">${this.escapeHtml(unc)}</span>
            </div>
        `).join('');
    }

    addToolCall(message) {
        const content = document.getElementById('tools-content');
        const status = document.getElementById('tools-status');

        // Update count
        status.textContent = this.toolCallCount;

        // Remove empty state if present
        const emptyState = content.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        // Create tool call element
        const toolEl = document.createElement('div');
        toolEl.className = 'tool-call-item new';
        toolEl.innerHTML = `
            <div class="tool-call-header">
                <span class="tool-name">${this.escapeHtml(message.tool_name)}</span>
                <span class="tool-time">${new Date(message.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="tool-call-args">${this.formatToolArgs(message.arguments)}</div>
            ${message.result_preview ? `<div class="tool-call-result">${this.escapeHtml(message.result_preview)}</div>` : ''}
        `;

        // Insert at top
        content.insertBefore(toolEl, content.firstChild);

        // Remove animation class after animation completes
        setTimeout(() => toolEl.classList.remove('new'), 500);

        // Limit number of tool calls shown
        while (content.children.length > 50) {
            content.removeChild(content.lastChild);
        }
    }

    formatToolArgs(args) {
        if (!args) return '';

        try {
            const entries = Object.entries(args);
            return entries.map(([key, value]) => {
                const displayValue = typeof value === 'string'
                    ? value.substring(0, 50) + (value.length > 50 ? '...' : '')
                    : JSON.stringify(value).substring(0, 50);
                return `<span class="arg-key">${key}:</span> <span class="arg-value">${this.escapeHtml(displayValue)}</span>`;
            }).join(', ');
        } catch (e) {
            return JSON.stringify(args).substring(0, 100);
        }
    }

    resetPanels() {
        document.getElementById('objectives-content').innerHTML = '<div class="loading">Initializing...</div>';
        document.getElementById('subtasks-content').innerHTML = '<div class="empty-state">No sub-tasks yet...</div>';
        document.getElementById('hypotheses-content').innerHTML = '<div class="empty-state">No hypotheses yet...</div>';
        document.getElementById('tools-content').innerHTML = '<div class="empty-state">No tool calls yet...</div>';
        document.getElementById('insights-content').innerHTML = '<div class="empty-state">No insights yet...</div>';
        document.getElementById('correlations-content').innerHTML = '<div class="empty-state">No correlations found...</div>';
        document.getElementById('reflections-content').innerHTML = '<div class="empty-state">No reflections yet...</div>';
        document.getElementById('uncertainties-content').innerHTML = '<div class="empty-state">No uncertainties identified...</div>';

        // Reset statuses
        document.getElementById('objectives-status').textContent = '--';
        document.getElementById('subtasks-status').textContent = '0';
        document.getElementById('hypotheses-status').textContent = '0';
        document.getElementById('tools-status').textContent = '0';
        document.getElementById('insights-status').textContent = '0';
        document.getElementById('correlations-status').textContent = '0';
        document.getElementById('reflections-status').textContent = '0';
        document.getElementById('uncertainties-status').textContent = '0';

        // Reset iteration
        document.getElementById('iteration-current').textContent = '0';
        document.getElementById('iteration-fill').style.width = '0%';
        document.getElementById('findings-count').textContent = '0';

        this.toolCallCount = 0;
    }

    // =========================================================================
    // Globe
    // =========================================================================

    initGlobe() {
        const container = document.getElementById('globe-container');
        if (!container) return;

        // Wait for container to have dimensions
        const rect = container.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) {
            setTimeout(() => this.initGlobe(), 100);
            return;
        }

        try {
            this.globe = Globe()
                .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
                .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
                .showAtmosphere(true)
                .atmosphereColor('#00ff41')
                .atmosphereAltitude(0.15)
                .pointOfView({ lat: 20, lng: 0, altitude: 2.5 })
                .pointColor(() => '#00ff41')
                .pointRadius(0.5)
                .pointAltitude(0.02)
                .pointsData([])
                .width(rect.width)
                .height(rect.height)
                (container);

            // Apply neon filter
            setTimeout(() => {
                const canvas = container.querySelector('canvas');
                if (canvas) {
                    canvas.style.filter = 'drop-shadow(0 0 15px rgba(0, 255, 65, 0.4))';
                }
            }, 500);

            // Handle resize
            window.addEventListener('resize', () => {
                if (this.globe && container) {
                    const newRect = container.getBoundingClientRect();
                    if (newRect.width > 0 && newRect.height > 0) {
                        this.globe.width(newRect.width);
                        this.globe.height(newRect.height);
                    }
                }
            });
        } catch (error) {
            console.error('Error initializing globe:', error);
        }
    }

    updateGlobeWithRegions(researchPlan) {
        if (!this.globe || !researchPlan) return;

        const regions = researchPlan.regions_of_interest || [];
        const status = document.getElementById('locations-status');

        if (regions.length === 0) {
            this.globe.pointsData([]);
            if (status) status.textContent = '--';
            return;
        }

        // Map common region names to coordinates
        const regionCoords = {
            'ukraine': { lat: 48.3794, lng: 31.1656 },
            'russia': { lat: 61.5240, lng: 105.3188 },
            'israel': { lat: 31.0461, lng: 34.8516 },
            'gaza': { lat: 31.3547, lng: 34.3088 },
            'iran': { lat: 32.4279, lng: 53.6880 },
            'syria': { lat: 34.8021, lng: 38.9968 },
            'china': { lat: 35.8617, lng: 104.1954 },
            'taiwan': { lat: 23.6978, lng: 120.9605 },
            'north korea': { lat: 40.3399, lng: 127.5101 },
            'south korea': { lat: 35.9078, lng: 127.7669 },
        };

        const points = regions.map(region => {
            const normalized = region.toLowerCase();
            const coords = regionCoords[normalized] || { lat: 0, lng: 0 };

            return {
                lat: coords.lat,
                lng: coords.lng,
                size: 0.8,
                color: '#00ff41',
                label: region,
            };
        }).filter(p => p.lat !== 0 || p.lng !== 0);

        this.globe.pointsData(points);
        this.globe.pointLabel('label');

        if (status) status.textContent = points.length;

        // Focus on first region if available
        if (points.length > 0) {
            this.globe.pointOfView({ lat: points[0].lat, lng: points[0].lng, altitude: 2 }, 1000);
        }
    }

    // =========================================================================
    // Event Listeners
    // =========================================================================

    initEventListeners() {
        // Start research button
        const startBtn = document.getElementById('start-research');
        const queryInput = document.getElementById('research-query');

        if (startBtn) {
            startBtn.addEventListener('click', () => this.startResearch());
        }

        if (queryInput) {
            queryInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.startResearch();
                }
            });
        }
    }

    async startResearch() {
        const queryInput = document.getElementById('research-query');
        const statusEl = document.getElementById('input-status');
        const startBtn = document.getElementById('start-research');

        const query = queryInput?.value?.trim();

        if (!query) {
            if (statusEl) {
                statusEl.textContent = 'Please enter a research query';
                statusEl.className = 'input-status error';
            }
            return;
        }

        // Disable button
        if (startBtn) {
            startBtn.disabled = true;
            startBtn.querySelector('.button-text').textContent = 'STARTING...';
        }

        if (statusEl) {
            statusEl.textContent = 'Starting research...';
            statusEl.className = 'input-status';
        }

        try {
            const response = await fetch('/api/agent/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, max_iterations: 5 }),
            });

            const result = await response.json();

            if (response.ok) {
                if (statusEl) {
                    statusEl.textContent = `Research started: ${result.session_id.substring(0, 8)}...`;
                    statusEl.className = 'input-status success';
                }

                // Clear input
                if (queryInput) queryInput.value = '';
            } else {
                throw new Error(result.detail || 'Failed to start research');
            }
        } catch (error) {
            console.error('Error starting research:', error);

            if (statusEl) {
                statusEl.textContent = `Error: ${error.message}`;
                statusEl.className = 'input-status error';
            }
        } finally {
            // Re-enable button
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.querySelector('.button-text').textContent = 'START';
            }
        }
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.agentBrain = new AgentBrainDashboard();
});
