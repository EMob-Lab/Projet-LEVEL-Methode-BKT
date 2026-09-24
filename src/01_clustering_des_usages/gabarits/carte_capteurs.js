<script>
/* Logique interactive de la carte des capteurs (voir carte.py::_js) : légende cliquable (isole un ou
plusieurs clusters), panneau à onglets (un onglet par échelle de profil demandée + un onglet
"marqueurs"), barre d'échelle maison, ajustement automatique du cadrage à l'ouverture. Les valeurs
__EN_MAJUSCULES__ sont remplacées par carte.py avant d'écrire le HTML final (voir _js()). */
const profileImages = __IMAGES__;
const colors = __COLORS__;
const names = __NAMES__;
const counts = __COUNTS__;
const clusterIds = __CLUSTER_IDS__;
const markerRecords = __MARKERS__;
const tabs = __TABS__;  // [{cle, titre}, ...] - un par échelle de profil réellement construite
const selectedClusters = new Set();

function renderProfileTab(tab) {
    const images = profileImages[tab];
    if (!images || Object.keys(images).length === 0) {
        document.getElementById('profiles-content').innerHTML = '<p>Aucun profil disponible.</p>';
        return;
    }
    let html = '';
    clusterIds.forEach(c => {
        if (!images[c]) return;
        html += `<div class="card"><div class="card-header"><span class="dot" style="background:${colors[c]};"></span>` +
            `<div class="card-title">${names[c] || c} <span class="card-meta">(n=${counts[c] || 0})</span></div></div>` +
            `<img class="card-plot" src="data:image/png;base64,${images[c]}"/></div>`;
    });
    document.getElementById('profiles-content').innerHTML = html;
}

function switchProfileTab(tab) {
    ['profiles-content', 'markers-content'].forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'none'; });
    if (tab === 'markers') document.getElementById('markers-content').style.display = 'block';
    else { document.getElementById('profiles-content').style.display = 'grid'; renderProfileTab(tab); }
    document.querySelectorAll('.legend-tab-btn').forEach(b => b.classList.remove('active-tab'));
    const active = document.getElementById('tab-' + tab);
    if (active) active.classList.add('active-tab');
}

function initTabs() {
    const container = document.getElementById('legend-tabs-container');
    let html = '';
    tabs.forEach(t => { html += `<button id="tab-${t.cle}" class="legend-tab-btn" onclick="switchProfileTab('${t.cle}')">${t.titre}</button>`; });
    html += `<button id="tab-markers" class="legend-tab-btn" onclick="switchProfileTab('markers')">Marqueurs</button>`;
    container.innerHTML = html;
}

function initClusterLegend() {
    let html = '', total = 0;
    clusterIds.forEach(c => {
        const cnt = counts[c] || 0;
        total += cnt;
        html += `<div class="cluster-row" data-cluster="${c}" role="button" tabindex="0" title="Cliquer pour filtrer"><span class="cluster-dot" style="background:${colors[c]};"></span><span class="cluster-label">${names[c] || c}</span><span class="cluster-count">${cnt}</span></div>`;
    });
    document.getElementById('cluster-counts').innerHTML = html;
    document.getElementById('total-count').textContent = total;
    document.querySelectorAll('.cluster-row').forEach((row) => {
        row.addEventListener('click', () => toggleClusterSelection(row.dataset.cluster));
        row.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); toggleClusterSelection(row.dataset.cluster); } });
    });
}

function updateLegendSelectionState() {
    document.querySelectorAll('.cluster-row').forEach((row) => row.classList.toggle('active-cluster', selectedClusters.has(row.dataset.cluster)));
}

function updateMarkerStyles() {
    const hasSelection = selectedClusters.size > 0;
    markerRecords.forEach((rec) => {
        const markerObj = window[rec.marker_name];
        if (!markerObj || typeof markerObj.getElement !== 'function') return;
        const el = markerObj.getElement();
        if (!el) return;
        const isHighlighted = rec.clusters.some((c) => selectedClusters.has(String(c)));
        el.classList.remove('marker-normal', 'marker-dimmed', 'marker-highlighted');
        if (!hasSelection) { el.classList.add('marker-normal'); markerObj.setZIndexOffset(0); }
        else if (isHighlighted) { el.classList.add('marker-highlighted'); markerObj.setZIndexOffset(1000); }
        else { el.classList.add('marker-dimmed'); markerObj.setZIndexOffset(-500); }
    });
}

function toggleClusterSelection(clusterId) {
    if (selectedClusters.has(clusterId)) selectedClusters.delete(clusterId); else selectedClusters.add(clusterId);
    updateLegendSelectionState();
    updateMarkerStyles();
}

function toggleCollapse() {
    const legend = document.getElementById('interactive-legend');
    const body = document.getElementById('legend-body');
    const btn = document.getElementById('btnCollapse');
    const collapsed = legend.classList.toggle('legend-collapsed');
    if (body) body.style.display = collapsed ? 'none' : 'flex';
    btn.textContent = collapsed ? 'Afficher' : 'Cacher';
}

function niceNum(x, round) {
    let exp = Math.floor(Math.log10(x));
    let f = x / Math.pow(10, exp);
    let nf;
    if (round) { if (f < 1.5) nf = 1; else if (f < 3) nf = 2; else if (f < 7) nf = 5; else nf = 10; }
    else { if (f <= 1) nf = 1; else if (f <= 2) nf = 2; else if (f <= 5) nf = 5; else nf = 10; }
    return nf * Math.pow(10, exp);
}

function updateScale() {
    const map = window['__MAPNAME__'];
    if (!map) return;
    const maxPx = 150;
    const center = map.getCenter();
    const centerPt = map.latLngToContainerPoint(center);
    const rightLatLng = map.containerPointToLatLng(centerPt.add([maxPx, 0]));
    const dist = center.distanceTo(rightLatLng);
    let nice = niceNum(dist, false);
    const niceR = niceNum(dist, true);
    nice = niceR <= dist ? niceR : nice;
    const barPx = Math.min(maxPx, maxPx * (nice / dist));
    const line = document.querySelector('.scale-line');
    if (line) line.style.width = barPx + 'px';
    const label = document.querySelector('.scale-label');
    if (label) label.textContent = nice >= 1000 ? (nice / 1000).toFixed(nice % 1000 === 0 ? 0 : 1) + ' km' : Math.round(nice) + ' m';
}

function createScaleBar() {
    if (document.getElementById('custom-scale-bar')) return;
    const div = document.createElement('div'); div.id = 'custom-scale-bar';
    const lc = document.createElement('div'); lc.className = 'scale-line-container';
    const line = document.createElement('div'); line.className = 'scale-line'; line.style.width = '100px';
    const lbl = document.createElement('div'); lbl.className = 'scale-label'; lbl.textContent = '...';
    lc.appendChild(line); div.appendChild(lc); div.appendChild(lbl);
    document.body.appendChild(div);
}

window.addEventListener('load', function () {
    createScaleBar();
    const map = window['__MAPNAME__'];
    if (map && typeof map.fitBounds === 'function') {
        let fitted = false;
        const fitFrance = () => {
            map.invalidateSize();
            const sz = map.getSize();
            if (fitted || sz.x < 50 || sz.y < 50) return;
            fitted = true;
            const wide = sz.x > 768;
            map.fitBounds(__BOUNDS__, {
                paddingTopLeft: [wide ? 290 : 10, 50],
                paddingBottomRight: [20, Math.min(340, Math.round(sz.y * 0.45))],
                animate: false
            });
            updateScale();
        };
        fitFrance();
        if (!fitted) new ResizeObserver(fitFrance).observe(map.getContainer());
    }
    updateScale();
    if (map && typeof map.on === 'function') {
        map.on('zoomend moveend', () => { updateScale(); updateMarkerStyles(); });
        map.on('layeradd', updateMarkerStyles);
    }
    initTabs();
    initClusterLegend();
    updateLegendSelectionState();
    updateMarkerStyles();
    switchProfileTab(tabs.length > 0 ? tabs[0].cle : 'markers');
    document.getElementById('btnCollapse').addEventListener('click', toggleCollapse);
});
</script>
