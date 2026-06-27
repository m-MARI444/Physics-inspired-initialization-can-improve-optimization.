import torch
import torch.nn.functional as F
import os
import sys
import json
import http.server
import socketserver
import urllib.parse
from transformers import GPT2TokenizerFast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pssa_gpt import PSSAGPT

# HTML/CSS/JS Single-Page Application for the visual dashboard
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PSSA-GPT Cognitive & Interpretability Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0b0e;
            --panel-bg: rgba(18, 18, 24, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00f0ff;
            --accent-magenta: #ff007f;
            --accent-purple: #9d4edd;
            --accent-green: #390099;
            --accent-yellow: #ffb703;
            --text-main: #f3f3f5;
            --text-muted: #8e8e93;
            --glow-cyan: 0 0 15px rgba(0, 240, 255, 0.45);
            --glow-magenta: 0 0 15px rgba(255, 0, 127, 0.45);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(157, 78, 221, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(0, 240, 255, 0.08) 0%, transparent 40%);
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 40px;
            border-bottom: 1px solid var(--border-color);
            backdrop-filter: blur(10px);
            background: rgba(11, 11, 14, 0.8);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo-section h1 {
            font-size: 24px;
            font-weight: 800;
            letter-spacing: 1px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo-section span {
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            padding: 3px 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            border: 1px solid var(--border-color);
            color: var(--accent-cyan);
            text-transform: uppercase;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #39ff14;
            border-radius: 50%;
            box-shadow: 0 0 8px #39ff14;
        }

        main {
            padding: 30px 40px;
            max-width: 1600px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 30px;
        }

        .control-panel {
            display: flex;
            flex-direction: column;
            gap: 25px;
        }

        .card {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: rgba(0, 240, 255, 0.15);
        }

        h2 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
            border-left: 3px solid var(--accent-cyan);
            padding-left: 10px;
        }

        textarea {
            width: 100%;
            height: 100px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            color: var(--text-main);
            padding: 12px;
            font-family: inherit;
            font-size: 14px;
            resize: none;
            margin-bottom: 15px;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: var(--accent-cyan);
        }

        button {
            width: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-magenta));
            border: none;
            color: white;
            padding: 14px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 15px rgba(255, 0, 127, 0.2);
        }

        button:hover {
            opacity: 0.9;
        }

        button:active {
            transform: scale(0.98);
        }

        /* Narrative Preset List */
        .preset-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 15px;
        }

        .preset-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .preset-item:hover {
            background: rgba(0, 240, 255, 0.05);
            border-color: rgba(0, 240, 255, 0.3);
        }

        /* Player Controls */
        .player-controls {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-top: 10px;
        }

        .btn-circle {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 18px;
            transition: all 0.2s;
        }

        .btn-circle:hover {
            background: rgba(0, 240, 255, 0.1);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            box-shadow: var(--glow-cyan);
        }

        .btn-circle.active {
            background: var(--accent-cyan);
            color: black;
            border-color: var(--accent-cyan);
        }

        /* Stepper progress */
        .step-progress-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
            width: 100%;
        }

        .step-slider {
            flex: 1;
            -webkit-appearance: none;
            background: rgba(255, 255, 255, 0.1);
            height: 6px;
            border-radius: 3px;
            outline: none;
        }

        .step-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
            box-shadow: var(--glow-cyan);
        }

        .step-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--accent-cyan);
        }

        /* Dashboard layouts */
        .dashboard-content {
            display: flex;
            flex-direction: column;
            gap: 30px;
        }

        /* Token stream visualizer */
        .token-stream {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            min-height: 50px;
            align-items: center;
        }

        .token-node {
            padding: 8px 14px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .token-node.passed {
            color: var(--text-muted);
            border-color: rgba(255, 255, 255, 0.05);
            background: rgba(255, 255, 255, 0.01);
        }

        .token-node.active {
            background: rgba(0, 240, 255, 0.15);
            border-color: var(--accent-cyan);
            color: white;
            box-shadow: var(--glow-cyan);
            font-weight: 600;
        }

        .token-node.upcoming {
            opacity: 0.5;
        }

        /* Workspace winner layout */
        .workspace-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 10px;
        }

        .workspace-item {
            border: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.01);
            border-radius: 10px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: all 0.3s ease;
            position: relative;
        }

        .workspace-item.active {
            border-color: var(--accent-yellow);
            background: rgba(255, 183, 3, 0.08);
            box-shadow: 0 0 15px rgba(255, 183, 3, 0.15);
        }

        .workspace-item.active .light {
            background-color: var(--accent-yellow);
            box-shadow: 0 0 10px var(--accent-yellow);
        }

        .workspace-item .light {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.08);
            position: absolute;
            top: 10px;
            right: 10px;
            transition: all 0.3s;
        }

        .workspace-item .name {
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
        }

        .workspace-item .val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
        }

        /* 12 Entity Bank Registers Grid */
        .registers-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }

        .register-box {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s;
        }

        .register-box.active-update {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.05);
        }

        .register-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--text-muted);
        }

        .register-box.scope-0::before { background: var(--accent-cyan); }
        .register-box.scope-1::before { background: var(--accent-magenta); }
        .register-box.scope-2::before { background: var(--accent-purple); }

        .register-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
        }

        .register-label {
            font-weight: 600;
            color: white;
        }

        .register-content {
            font-size: 15px;
            font-weight: 600;
            color: var(--accent-cyan);
            min-height: 22px;
        }

        .register-timeline {
            display: flex;
            flex-direction: column;
            gap: 4px;
            border-top: 1px solid rgba(255,255,255,0.05);
            padding-top: 8px;
            font-size: 11px;
        }

        .timeline-row {
            display: flex;
            justify-content: space-between;
            color: var(--text-muted);
        }

        .timeline-row span:last-child {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
        }

        /* Heatmaps Matrix layout */
        .matrix-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        .matrix-wrapper {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .matrix-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 3px;
            aspect-ratio: 1;
            background: rgba(255,255,255,0.02);
            padding: 8px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .matrix-cell {
            background-color: rgba(255, 255, 255, 0.02);
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }

        .matrix-cell:hover {
            outline: 1px solid white;
            z-index: 10;
        }

        .cell-tooltip {
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #1c1c24;
            border: 1px solid var(--border-color);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
            display: none;
            z-index: 100;
            pointer-events: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }

        .matrix-cell:hover .cell-tooltip {
            display: block;
        }

        /* Sparkline chart */
        .chart-container {
            width: 100%;
            height: 120px;
            background: rgba(0,0,0,0.2);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 10px;
            display: flex;
            align-items: center;
        }

        svg.saliency-chart {
            width: 100%;
            height: 100%;
            overflow: visible;
        }

        /* Metrics details */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }

        .metric-box {
            display: flex;
            flex-direction: column;
            gap: 4px;
            background: rgba(0,0,0,0.15);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px 16px;
        }

        .metric-box .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .metric-box .val {
            font-size: 20px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .metric-box .val.alert {
            color: var(--accent-magenta);
        }

        .hidden {
            display: none !important;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-section">
            <h1>PSSA-GPT <span>Cognitive Dashboard</span></h1>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            <span>MODEL: PSSA-GPT (50M, Step 25K)</span>
        </div>
    </header>

    <main>
        <!-- Left Sidebar: Controls & Inputs -->
        <section class="control-panel">
            <div class="card">
                <h2>Input Story / Text</h2>
                <textarea id="story-input">John bought a red bicycle yesterday. He is happy.</textarea>
                <button id="run-btn">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M11.596 8.697l-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.233-.696l6.363 3.692a.802.802 0 0 1 0 1.393z"/>
                    </svg>
                    Analyze Cognitive State
                </button>

                <div class="preset-list">
                    <div class="preset-item" onclick="loadPreset('John bought a red bicycle yesterday.')">
                        "John bought a red bicycle yesterday."
                    </div>
                    <div class="preset-item" onclick="loadPreset('Lily saw a small bird. Spot saw a big dog. She said the dog was happy.')">
                        "Lily saw a small bird. Spot saw a big dog..."
                    </div>
                    <div class="preset-item" onclick="loadPreset('Spot saw Kitty. Actually, it was Tim. He felt very happy.')">
                        "Spot saw Kitty. Actually, it was Tim..."
                    </div>
                </div>
            </div>

            <div class="card player-card">
                <h2>Playback Controls</h2>
                <div class="player-controls">
                    <button class="btn-circle" id="prev-btn" onclick="prevStep()">◀</button>
                    <button class="btn-circle" id="play-btn" onclick="togglePlay()">▶</button>
                    <button class="btn-circle" id="next-btn" onclick="nextStep()">▶</button>
                </div>
                <div class="step-progress-container">
                    <input type="range" min="0" max="0" value="0" class="step-slider" id="step-slider" oninput="onSliderChange(this.value)">
                    <span class="step-label" id="step-label">0/0</span>
                </div>
            </div>

            <div class="card">
                <h2>Legend / Terminology</h2>
                <div style="font-size: 13px; display: flex; flex-direction: column; gap: 8px; color: var(--text-muted);">
                    <div><b style="color:var(--accent-cyan)">S0 (Perception)</b>: Slot tracks external surprise cues.</div>
                    <div><b style="color:var(--accent-magenta)">S1 (Memory)</b>: Slot tracks counterfactual histories.</div>
                    <div><b style="color:var(--accent-purple)">S2 (Planning)</b>: Slot tracks prospective simulated futures.</div>
                    <div><b style="color:var(--accent-yellow)">Ignition</b>: Momentary consciousness sync from T0 into timelines.</div>
                </div>
            </div>
        </section>

        <!-- Right Main View: Visualization panels -->
        <section class="dashboard-content">
            <!-- Token Stream View -->
            <div class="card">
                <h2>BPE Token Stream & Saliency Overlay</h2>
                <div class="token-stream" id="token-stream">
                    <!-- Tokens will be dynamically populated -->
                </div>
            </div>

            <!-- Workspace Winners / Cognitive Modules -->
            <div class="card">
                <h2>Global Workspace Competition (Conscious Winner)</h2>
                <div class="workspace-grid" id="workspace-grid">
                    <!-- Workspace items -->
                    <div class="workspace-item" id="ws-0"><div class="light"></div><div class="name">Perception</div><div class="val" id="ws-v-0">0.00</div></div>
                    <div class="workspace-item" id="ws-1"><div class="light"></div><div class="name">Affordance</div><div class="val" id="ws-v-1">0.00</div></div>
                    <div class="workspace-item" id="ws-2"><div class="light"></div><div class="name">Language</div><div class="val" id="ws-v-2">0.00</div></div>
                    <div class="workspace-item" id="ws-3"><div class="light"></div><div class="name">Memory</div><div class="val" id="ws-v-3">0.00</div></div>
                    <div class="workspace-item" id="ws-4"><div class="light"></div><div class="name">Planning</div><div class="val" id="ws-v-4">0.00</div></div>
                    <div class="workspace-item" id="ws-5"><div class="light"></div><div class="name">Affective</div><div class="val" id="ws-v-5">0.00</div></div>
                    <div class="workspace-item" id="ws-6"><div class="light"></div><div class="name">Executive</div><div class="val" id="ws-v-6">0.00</div></div>
                    <div class="workspace-item" id="ws-7"><div class="light"></div><div class="name">Meta-Cognitive</div><div class="val" id="ws-v-7">0.00</div></div>
                </div>
            </div>

            <!-- Scientific Layer Metrics -->
            <div class="card">
                <h2>Thermodynamic Metrics (Step State)</h2>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <span class="label">Cognitive Drift</span>
                        <span class="val" id="metric-drift">0.0000</span>
                    </div>
                    <div class="metric-box">
                        <span class="label">Paradigm Pressure</span>
                        <span class="val" id="metric-paradigm">0.0000</span>
                    </div>
                    <div class="metric-box">
                        <span class="label">Introspection Fatigue</span>
                        <span class="val" id="metric-fatigue">0.0000</span>
                    </div>
                    <div class="metric-box">
                        <span class="label">Conscious Ignition?</span>
                        <span class="val" id="metric-ignition">No</span>
                    </div>
                </div>
            </div>

            <!-- 12 Entity Bank Registers Grid -->
            <div class="card">
                <h2>12 Persistent Entity Bank Registers & Multi-Timeline Versioning</h2>
                <div class="registers-grid" id="registers-grid">
                    <!-- 12 boxes (3 scopes * 4 banks) will be generated here -->
                </div>
            </div>

            <!-- Matrices heatmaps -->
            <div class="matrix-container">
                <!-- Directed Causal Dependency Matrix -->
                <div class="card matrix-wrapper">
                    <h2>Directed Causal Dependency Matrix (D)</h2>
                    <div class="matrix-grid" id="causal-matrix">
                        <!-- 12x12 grid cell -->
                    </div>
                </div>

                <!-- Cross-Timeline Causal Attribution Matrix -->
                <div class="card matrix-wrapper">
                    <h2>Cross-Timeline Attribution Matrix (A_cf)</h2>
                    <div class="matrix-grid" id="cf-matrix">
                        <!-- 12x12 grid cell -->
                    </div>
                </div>
            </div>

            <!-- SVG Saliency chart -->
            <div class="card">
                <h2>Emergent 3-Scope Attention Chart (Layer 2)</h2>
                <div class="chart-container">
                    <svg class="saliency-chart" id="saliency-chart" viewBox="0 0 1000 100">
                        <path id="path-s0" fill="none" stroke="#00f0ff" stroke-width="2.5"></path>
                        <path id="path-s1" fill="none" stroke="#ff007f" stroke-width="2.5"></path>
                        <path id="path-s2" fill="none" stroke="#9d4edd" stroke-width="2.5"></path>
                        <line id="slider-indicator" x1="0" y1="0" x2="0" y2="100" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" stroke-dasharray="4"></line>
                    </svg>
                </div>
            </div>
        </section>
    </main>

    <script>
        let dataTrace = null;
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;

        const moduleColors = {
            "Perc": "#00f0ff",
            "Afford": "#390099",
            "Lang": "#39ff14",
            "Mem": "#ff007f",
            "Plan": "#9d4edd",
            "Affec": "#ffb703",
            "Exec": "#ff5400",
            "Meta": "#e0aaff"
        };

        // Initialize 12 boxes
        const registersGrid = document.getElementById('registers-grid');
        for (let s = 0; s < 3; s++) {
            for (let b = 0; b < 4; b++) {
                const box = document.createElement('div');
                box.className = `register-box scope-${s}`;
                box.id = `reg-${s}-${b}`;
                box.innerHTML = `
                    <div class="register-header">
                        <span class="register-label">SCOPE ${s} / BANK ${b}</span>
                    </div>
                    <div class="register-content" id="reg-c-${s}-${b}">empty</div>
                    <div class="register-timeline">
                        <div class="timeline-row"><span>T0 (Active):</span><span id="t0-${s}-${b}">empty</span></div>
                        <div class="timeline-row"><span>T1 (Hist):</span><span id="t1-${s}-${b}">empty</span></div>
                        <div class="timeline-row"><span>T2 (Prosp):</span><span id="t2-${s}-${b}">empty</span></div>
                    </div>
                `;
                registersGrid.appendChild(box);
            }
        }

        // Initialize 12x12 matrices
        const causalMatrix = document.getElementById('causal-matrix');
        const cfMatrix = document.getElementById('cf-matrix');
        for (let i = 0; i < 144; i++) {
            causalMatrix.appendChild(createCell(i));
            cfMatrix.appendChild(createCell(i));
        }

        function createCell(i) {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.id = `cell-${i}`;
            const tooltip = document.createElement('div');
            tooltip.className = 'cell-tooltip';
            tooltip.id = `tooltip-${i}`;
            cell.appendChild(tooltip);
            return cell;
        }

        function loadPreset(text) {
            document.getElementById('story-input').value = text;
            runAnalysis();
        }

        async function runAnalysis() {
            const text = document.getElementById('story-input').value;
            const runBtn = document.getElementById('run-btn');
            runBtn.disabled = true;
            runBtn.innerText = "Analyzing Model State...";

            try {
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const resData = await response.json();
                
                if (resData.error) {
                    alert("Error running analysis: " + resData.error);
                } else {
                    dataTrace = resData.steps;
                    currentStep = 0;
                    isPlaying = false;
                    clearInterval(playInterval);
                    document.getElementById('play-btn').innerText = "▶";

                    // Update UI controls
                    const slider = document.getElementById('step-slider');
                    slider.max = dataTrace.length - 1;
                    slider.value = 0;
                    document.getElementById('step-label').innerText = `1/${dataTrace.length}`;

                    // Update layouts
                    renderTokenStream();
                    renderSaliencyChart();
                    updateStepView();
                }
            } catch (err) {
                alert("Network error: " + err);
            } finally {
                runBtn.disabled = false;
                runBtn.innerText = "Analyze Cognitive State";
            }
        }

        function renderTokenStream() {
            const container = document.getElementById('token-stream');
            container.innerHTML = '';
            dataTrace.forEach((step, idx) => {
                const node = document.createElement('span');
                node.className = 'token-node';
                node.id = `tok-${idx}`;
                node.innerText = step.token.replace('Ġ', ' ');
                node.onclick = () => {
                    currentStep = idx;
                    document.getElementById('step-slider').value = idx;
                    updateStepView();
                };
                container.appendChild(node);
            });
        }

        function renderSaliencyChart() {
            if (!dataTrace) return;
            const width = 1000;
            const height = 100;
            const pointsCount = dataTrace.length;
            const xStep = width / Math.max(1, pointsCount - 1);

            let s0Pts = [], s1Pts = [], s2Pts = [];
            dataTrace.forEach((step, idx) => {
                const x = idx * xStep;
                s0Pts.push(`${x},${height - step.scopes[0] * height}`);
                s1Pts.push(`${x},${height - step.scopes[1] * height}`);
                s2Pts.push(`${x},${height - step.scopes[2] * height}`);
            });

            document.getElementById('path-s0').setAttribute('d', 'M' + s0Pts.join(' L'));
            document.getElementById('path-s1').setAttribute('d', 'M' + s1Pts.join(' L'));
            document.getElementById('path-s2').setAttribute('d', 'M' + s2Pts.join(' L'));
        }

        function updateStepView() {
            if (!dataTrace) return;
            const stepData = dataTrace[currentStep];
            
            // Highlight tokens
            dataTrace.forEach((_, idx) => {
                const node = document.getElementById(`tok-${idx}`);
                if (!node) return;
                node.className = 'token-node';
                if (idx < currentStep) node.classList.add('passed');
                else if (idx === currentStep) node.classList.add('active');
                else node.classList.add('upcoming');
            });

            // Update step labels
            document.getElementById('step-label').innerText = `${currentStep + 1}/${dataTrace.length}`;

            // Update Saliency Chart Indicator Line
            const width = 1000;
            const xStep = width / Math.max(1, dataTrace.length - 1);
            document.getElementById('slider-indicator').setAttribute('x1', currentStep * xStep);
            document.getElementById('slider-indicator').setAttribute('x2', currentStep * xStep);

            // Update Workspace Winner
            const moduleNames = ["Perc", "Afford", "Lang", "Mem", "Plan", "Affec", "Exec", "Meta"];
            for (let m = 0; m < 8; m++) {
                const item = document.getElementById(`ws-${m}`);
                const valLabel = document.getElementById(`ws-v-${m}`);
                const activeVal = stepData.saliences[m];
                valLabel.innerText = activeVal.toFixed(3);
                item.className = 'workspace-item';
                if (m === stepData.winner) {
                    item.classList.add('active');
                }
            }

            // Update metrics
            document.getElementById('metric-drift').innerText = stepData.drift.toFixed(4);
            document.getElementById('metric-paradigm').innerText = stepData.paradigm.toFixed(4);
            document.getElementById('metric-fatigue').innerText = stepData.fatigue.toFixed(4);
            const ignitionVal = document.getElementById('metric-ignition');
            if (stepData.ignition) {
                ignitionVal.innerText = "YES (Consensus)";
                ignitionVal.classList.add('alert');
            } else {
                ignitionVal.innerText = "No";
                ignitionVal.classList.remove('alert');
            }

            // Update Registers
            for (let s = 0; s < 3; s++) {
                for (let b = 0; b < 4; b++) {
                    const flatIdx = s * 4 + b;
                    const rData = stepData.registers[flatIdx];
                    const box = document.getElementById(`reg-${s}-${b}`);
                    
                    document.getElementById(`reg-c-${s}-${b}`).innerText = rData.closest;
                    document.getElementById(`t0-${s}-${b}`).innerText = rData.t0;
                    document.getElementById(`t1-${s}-${b}`).innerText = rData.t1;
                    document.getElementById(`t2-${s}-${b}`).innerText = rData.t2;

                    box.className = `register-box scope-${s}`;
                    if (rData.was_updated) {
                        box.classList.add('active-update');
                    }
                }
            }

            // Update Matrices Heatmaps
            const causalMatrix = document.getElementById('causal-matrix');
            const cfMatrix = document.getElementById('cf-matrix');
            
            for (let row = 0; row < 12; row++) {
                for (let col = 0; col < 12; col++) {
                    const flatIdx = row * 12 + col;
                    
                    // Causal Matrix Cell
                    const cellC = causalMatrix.children[flatIdx];
                    const valC = stepData.causal[row][col];
                    cellC.style.backgroundColor = `rgba(0, 240, 255, ${valC})`;
                    cellC.children[0].innerText = `Scope ${Math.floor(row/4)} Bank ${row%4} -> Scope ${Math.floor(col/4)} Bank ${col%4}: ${valC.toFixed(4)}`;

                    // CF Matrix Cell
                    const cellCF = cfMatrix.children[flatIdx];
                    const valCF = stepData.cf[row][col];
                    cellCF.style.backgroundColor = `rgba(255, 0, 127, ${valCF})`;
                    cellCF.children[0].innerText = `Active R${row} -> Hist R${col}: ${valCF.toFixed(4)}`;
                }
            }
        }

        function togglePlay() {
            isPlaying = !isPlaying;
            const playBtn = document.getElementById('play-btn');
            if (isPlaying) {
                playBtn.innerText = "⏸";
                playInterval = setInterval(() => {
                    if (currentStep < dataTrace.length - 1) {
                        currentStep++;
                        document.getElementById('step-slider').value = currentStep;
                        updateStepView();
                    } else {
                        currentStep = 0;
                        document.getElementById('step-slider').value = 0;
                        updateStepView();
                    }
                }, 1500);
            } else {
                playBtn.innerText = "▶";
                clearInterval(playInterval);
            }
        }

        function prevStep() {
            if (currentStep > 0) {
                currentStep--;
                document.getElementById('step-slider').value = currentStep;
                updateStepView();
            }
        }

        function nextStep() {
            if (currentStep < dataTrace.length - 1) {
                currentStep++;
                document.getElementById('step-slider').value = currentStep;
                updateStepView();
            }
        }

        function onSliderChange(val) {
            currentStep = parseInt(val);
            updateStepView();
        }

        document.getElementById('run-btn').onclick = runAnalysis;

        // Auto run initial text on start
        setTimeout(runAnalysis, 1000);
    </script>
</body>
</html>
"""

class VisualDashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url_parsed = urllib.parse.urlparse(self.path)
        if url_parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        elif url_parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        url_parsed = urllib.parse.urlparse(self.path)
        if url_parsed.path == "/api/run":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                params = json.loads(post_data.decode("utf-8"))
                text = params.get("text", "John bought a red bicycle yesterday.")
                
                # Execute inference and extract details
                steps_data = run_model_inference(text)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"steps": steps_data}).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

# Global reference to model and tokenizer
model = None
tokenizer = None
device = torch.device("cpu")

def run_model_inference(text):
    global model, tokenizer
    raw_tokens = tokenizer.encode(text)
    decoded_tokens = [tokenizer.decode([t]) for t in raw_tokens]
    
    tokens_tensor = torch.tensor([raw_tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits, gates, slots, pre_wave, adj, entities, retrievals, write_candidates, scope_weights, recon_loss, out_trust, out_recency, out_ignition, out_fatigue, out_drift, out_trust_div, out_paradigm = model(
            tokens_tensor
        )
        
    steps = []
    seq_len = len(raw_tokens)
    
    # Vocabulary weights mapping for semantic decoding
    embed_weights = model.embed.weight.detach()
    
    for t in range(seq_len):
        # Scope weights at Layer 2 (top layer)
        scopes = scope_weights[0, t, 2, :].tolist()
        
        # Winner module at Layer 2
        winner = model.last_workspace_winners[0, t, 2].item()
        
        # Saliences at Layer 2
        saliences = model.last_module_saliences[0, t, 2].tolist()
        
        # Ignition
        ignition = model.last_ignitions[0, t, 2].item()
        
        # Metrics
        drift = out_drift[0, t, 2].item()
        paradigm = out_paradigm[0, t, 2].item()
        fatigue = out_fatigue[0, t, 2].item()
        
        # Entity registers mapping closest tokens
        registers = []
        E_t = entities[0, t, 2, :, :, :] # [3, 4, 256]
        E_t_flat = E_t.view(12, 256)
        
        # Timeline versions mapping
        timeline_states = model.last_timeline_states[0, t, 2, :, :, :, :] # [3, 4, 3, 256]
        
        # Check active update by checking write intensity and allocation routing
        was_updated_list = [False] * 12
        if write_candidates is not None:
            u_t = write_candidates[0, t, 2, 2, :] # Entity slot update vector [256]
            u_t_norm = u_t.norm().item()
            if u_t_norm > 0.1:
                similarity = torch.matmul(u_t, E_t_flat.t()) / (256 ** 0.5)
                alloc_soft = F.softmax(similarity / 0.15, dim=-1)
                for idx in range(12):
                    was_updated_list[idx] = alloc_soft[idx].item() > 0.08
        
        for idx in range(12):
            s = idx // 4
            b = idx % 4
            
            # Map closest active concept
            emb = E_t_flat[idx]
            sims = F.cosine_similarity(emb.unsqueeze(0), embed_weights, dim=-1)
            closest_idx = torch.argmax(sims).item()
            closest_token = tokenizer.decode([closest_idx]).strip()
            if closest_token == "":
                closest_token = f"T_{closest_idx}"
                
            # Timeline 0
            sims_0 = F.cosine_similarity(timeline_states[s, b, 0, :].unsqueeze(0), embed_weights, dim=-1)
            t0_tok = tokenizer.decode([torch.argmax(sims_0).item()]).strip()
            
            # Timeline 1
            sims_1 = F.cosine_similarity(timeline_states[s, b, 1, :].unsqueeze(0), embed_weights, dim=-1)
            t1_tok = tokenizer.decode([torch.argmax(sims_1).item()]).strip()
            
            # Timeline 2
            sims_2 = F.cosine_similarity(timeline_states[s, b, 2, :].unsqueeze(0), embed_weights, dim=-1)
            t2_tok = tokenizer.decode([torch.argmax(sims_2).item()]).strip()
            
            was_updated = was_updated_list[idx]
            
            registers.append({
                "closest": closest_token,
                "t0": t0_tok if t0_tok != "" else "empty",
                "t1": t1_tok if t1_tok != "" else "empty",
                "t2": t2_tok if t2_tok != "" else "empty",
                "was_updated": was_updated
            })
            
        # Directed Causal Dependency Matrix
        Q_dep = model.layers[2].wq_dep(E_t_flat)
        K_dep = model.layers[2].wk_dep(E_t_flat)
        D_logits = torch.matmul(Q_dep, K_dep.transpose(0, 1)) / (16 ** 0.5)
        causal = torch.sigmoid(D_logits).tolist()
        
        # Cross-Timeline Causal Attribution Matrix A_cf
        cf_attn = model.layers[2].last_cf_attn[0].tolist() if hasattr(model.layers[2], 'last_cf_attn') else [[0.0]*12]*12
        
        steps.append({
            "token": decoded_tokens[t],
            "scopes": scopes,
            "winner": winner,
            "saliences": saliences,
            "ignition": ignition,
            "drift": drift,
            "paradigm": paradigm,
            "fatigue": fatigue,
            "registers": registers,
            "causal": causal,
            "cf": cf_attn
        })
        
    return steps

def run_dashboard_server():
    global model, tokenizer
    print("Loading tokenizer and 25K model checkpoint...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    vocab_size = len(tokenizer)
    d_model = 256
    num_layers = 6
    num_slots = 8
    
    model = PSSAGPT(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_slots=num_slots, 
        tau=0.15, 
        num_scopes=3, 
        num_layers=num_layers
    ).to(device)
    
    checkpoint_path = "checkpoints/pssa_campaign_latest.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    print(f"Loading weights from {checkpoint_path}...")
    pre_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(pre_ckpt["model_state_dict"])
    model.eval()
    print("Model loaded successfully!")
    
    # Find an available port starting at 8000
    port = 8000
    while port < 8100:
        try:
            handler = VisualDashboardHandler
            with socketserver.TCPServer(("", port), handler) as httpd:
                print(f"==========================================================")
                print(f"  PSSA-GPT COGNITIVE DASHBOARD RUNNING AT:")
                print(f"  --> http://localhost:{port}/")
                print(f"==========================================================")
                print("Press Ctrl+C to terminate.")
                httpd.serve_forever()
        except OSError:
            port += 1

if __name__ == "__main__":
    try:
        run_dashboard_server()
    except KeyboardInterrupt:
        print("\nDashboard server terminated.")
