// Global state
let currentEpisode = null;
let currentEpisodeIndex = -1;
let sortedEpisodes = [];
let currentAnnotations = [];
let originalAnnotations = [];
let planChoices = [];
let turningPoints = []; // List of {frame_index, subtask_index} for turning points

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    loadEpisodes();
    
    document.getElementById('episodeSelect').addEventListener('change', function() {
        const episodeKey = this.value;
        if (episodeKey) {
            loadEpisode(episodeKey);
        } else {
            hideEpisodeData();
        }
    });
    
    document.getElementById('prevEpisodeBtn').addEventListener('click', function() {
        if (currentEpisodeIndex > 0) {
            const prevEpisode = sortedEpisodes[currentEpisodeIndex - 1];
            document.getElementById('episodeSelect').value = prevEpisode.episode_key;
            loadEpisode(prevEpisode.episode_key);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    });
    
    document.getElementById('nextEpisodeBtn').addEventListener('click', function() {
        if (currentEpisodeIndex >= 0 && currentEpisodeIndex < sortedEpisodes.length - 1) {
            const nextEpisode = sortedEpisodes[currentEpisodeIndex + 1];
            document.getElementById('episodeSelect').value = nextEpisode.episode_key;
            loadEpisode(nextEpisode.episode_key);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    });
    
    document.getElementById('saveBtn').addEventListener('click', saveChanges);
});

// Load list of episodes
async function loadEpisodes() {
    try {
        const response = await fetch('/api/episodes');
        const data = await response.json();
        
        // Sort episodes by episode_key
        sortedEpisodes = data.episodes.sort((a, b) => {
            return a.episode_key.localeCompare(b.episode_key);
        });
        
        const select = document.getElementById('episodeSelect');
        select.innerHTML = '<option value="">Select an episode...</option>';
        
        sortedEpisodes.forEach(episode => {
            const option = document.createElement('option');
            option.value = episode.episode_key;
            option.textContent = `${episode.episode_key} (${episode.num_frames} frames, ${episode.num_keyframes} keyframes)`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading episodes:', error);
        document.getElementById('episodeSelect').innerHTML = '<option value="">Error loading episodes</option>';
    }
}

// Load episode data
async function loadEpisode(episodeKey) {
    try {
        const response = await fetch(`/api/episode/${episodeKey}`);
        if (!response.ok) {
            throw new Error('Failed to load episode');
        }
        
        const episode = await response.json();
        currentEpisode = episode;
        
        // Find current episode index in sorted list
        currentEpisodeIndex = sortedEpisodes.findIndex(ep => ep.episode_key === episodeKey);
        
        // Update navigation buttons
        updateNavigationButtons();
        
        // Store original annotations
        originalAnnotations = JSON.parse(JSON.stringify(episode.vlm_annotations || []));
        currentAnnotations = JSON.parse(JSON.stringify(episode.vlm_annotations || []));
        planChoices = episode.plan || [];
        
        // Initialize turning points from existing annotations
        // Extract turning points: frames where subtask changes
        turningPoints = extractTurningPointsFromAnnotations(
            currentAnnotations,
            episode.keyframes || []
        );
        
        // Display episode info
        displayEpisodeInfo(episode);
        
        // Display plan
        displayPlan(planChoices);
        
        // Load and display keyframes
        await displayKeyframes(episode);
        
        // Show save section
        document.getElementById('saveSection').style.display = 'block';
        document.getElementById('saveStatus').textContent = '';
        document.getElementById('saveStatus').className = 'save-status';
        
    } catch (error) {
        console.error('Error loading episode:', error);
        showError('Failed to load episode: ' + error.message);
    }
}

// Update navigation buttons state
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prevEpisodeBtn');
    const nextBtn = document.getElementById('nextEpisodeBtn');
    
    if (currentEpisodeIndex < 0) {
        prevBtn.disabled = true;
        nextBtn.disabled = true;
    } else {
        prevBtn.disabled = currentEpisodeIndex === 0;
        nextBtn.disabled = currentEpisodeIndex >= sortedEpisodes.length - 1;
    }
}

// Display episode information
function displayEpisodeInfo(episode) {
    document.getElementById('episodeTitle').textContent = `Episode: ${episode.episode_key}`;
    document.getElementById('instruction').textContent = episode.instruction || 'N/A';
    document.getElementById('numFrames').textContent = episode.num_frames || 0;
    document.getElementById('numKeyframes').textContent = episode.keyframes ? episode.keyframes.length : 0;
    document.getElementById('numSubtasks').textContent = episode.num_subtasks || 0;
    document.getElementById('episodeInfo').style.display = 'block';
}

// Display plan choices
function displayPlan(plan) {
    const planList = document.getElementById('planList');
    planList.innerHTML = '';
    
    if (plan.length === 0) {
        planList.innerHTML = '<div class="plan-item">No plan available</div>';
        return;
    }
    
    plan.forEach((item, index) => {
        const planItem = document.createElement('div');
        planItem.className = 'plan-item';
        planItem.innerHTML = `<strong>${String.fromCharCode(65 + index)}:</strong> ${item}`;
        planList.appendChild(planItem);
    });
    
    document.getElementById('planSection').style.display = 'block';
}

// Display keyframes
async function displayKeyframes(episode) {
    const container = document.getElementById('keyframesContainer');
    const grid = document.getElementById('keyframesGrid');
    grid.innerHTML = '<div class="loading">Loading keyframes...</div>';
    container.style.display = 'block';
    
    const keyframes = episode.keyframes || [];
    
    // Load all keyframe images
    const keyframePromises = keyframes.map(async (frameIdx) => {
        try {
            const response = await fetch(`/api/frame/${episode.episode_key}/${frameIdx}`);
            if (!response.ok) {
                throw new Error(`Failed to load frame ${frameIdx}`);
            }
            const data = await response.json();
            return { frameIdx, image: data.image };
        } catch (error) {
            console.error(`Error loading frame ${frameIdx}:`, error);
            return { frameIdx, image: null };
        }
    });
    
    const keyframeData = await Promise.all(keyframePromises);
    
    // Clear loading message
    grid.innerHTML = '';
    
    // Create keyframe items
    keyframeData.forEach(({ frameIdx, image }) => {
        const keyframeItem = createKeyframeItem(frameIdx, image, episode);
        grid.appendChild(keyframeItem);
    });
    
    // Update annotations from turning points after display
    if (turningPoints.length > 0) {
        currentAnnotations = generateAnnotationsFromTurningPoints(
            turningPoints,
            keyframes,
            episode.num_frames || 0,
            episode.num_subtasks || 1
        );
    }
}

// Extract turning points from annotations
function extractTurningPointsFromAnnotations(annotations, keyframes) {
    if (!annotations || annotations.length === 0) {
        return [];
    }
    
    // Sort annotations by frame_index
    const sorted = [...annotations].sort((a, b) => a.frame_index - b.frame_index);
    const turningPoints = [];
    
    // First frame is always a turning point
    if (sorted.length > 0) {
        turningPoints.push({
            frame_index: sorted[0].frame_index,
            subtask_index: sorted[0].subtask_index
        });
    }
    
    // Find frames where subtask changes
    for (let i = 1; i < sorted.length; i++) {
        if (sorted[i].subtask_index !== sorted[i - 1].subtask_index) {
            turningPoints.push({
                frame_index: sorted[i].frame_index,
                subtask_index: sorted[i].subtask_index
            });
        }
    }
    
    return turningPoints;
}

// Generate annotations from turning points
function generateAnnotationsFromTurningPoints(turningPoints, keyframes, numFrames, numSubtasks) {
    if (!turningPoints || turningPoints.length === 0) {
        // No turning points, assign all to subtask 1
        return keyframes.map(kf => ({ frame_index: kf, subtask_index: 1 }));
    }
    
    // Sort turning points by frame_index
    const sorted = [...turningPoints].sort((a, b) => a.frame_index - b.frame_index);
    
    // Generate full sequence
    const fullSequence = new Array(numFrames).fill(-1);
    
    // Assign subtasks based on turning points
    for (let i = 0; i < sorted.length; i++) {
        const tp = sorted[i];
        const startFrame = tp.frame_index;
        const subtaskIdx = tp.subtask_index;
        
        // Determine end frame
        const endFrame = (i + 1 < sorted.length) 
            ? sorted[i + 1].frame_index 
            : numFrames;
        
        // Assign this subtask to all frames in range
        for (let frameNum = startFrame; frameNum < endFrame && frameNum < numFrames; frameNum++) {
            fullSequence[frameNum] = subtaskIdx;
        }
    }
    
    // Fill any remaining gaps
    for (let i = 0; i < numFrames; i++) {
        if (fullSequence[i] === -1) {
            // Find previous non-empty value
            for (let j = i - 1; j >= 0; j--) {
                if (fullSequence[j] !== -1) {
                    fullSequence[i] = fullSequence[j];
                    break;
                }
            }
            if (fullSequence[i] === -1) {
                fullSequence[i] = 1;
            }
        }
    }
    
    // Generate annotations for all keyframes
    return keyframes.map(kf => ({
        frame_index: kf,
        subtask_index: fullSequence[kf] || 1
    }));
}

// Check if a frame is a turning point
function isTurningPoint(frameIdx) {
    return turningPoints.some(tp => tp.frame_index === frameIdx);
}

// Get inferred subtask for a frame based on turning points
function getInferredSubtask(frameIdx, keyframes, numFrames) {
    if (turningPoints.length === 0) {
        return 1;
    }
    
    const sorted = [...turningPoints].sort((a, b) => a.frame_index - b.frame_index);
    
    // Find which turning point range this frame falls into
    for (let i = 0; i < sorted.length; i++) {
        const tp = sorted[i];
        const startFrame = tp.frame_index;
        const endFrame = (i + 1 < sorted.length) 
            ? sorted[i + 1].frame_index 
            : numFrames;
        
        if (frameIdx >= startFrame && frameIdx < endFrame) {
            return tp.subtask_index;
        }
    }
    
    // Fallback
    return sorted[0].subtask_index;
}

// Add or update a turning point
function addOrUpdateTurningPoint(frameIdx, subtaskIndex) {
    const existingIndex = turningPoints.findIndex(tp => tp.frame_index === frameIdx);
    if (existingIndex >= 0) {
        turningPoints[existingIndex].subtask_index = subtaskIndex;
    } else {
        turningPoints.push({
            frame_index: frameIdx,
            subtask_index: subtaskIndex
        });
    }
}

// Remove a turning point
function removeTurningPoint(frameIdx) {
    const index = turningPoints.findIndex(tp => tp.frame_index === frameIdx);
    if (index >= 0) {
        turningPoints.splice(index, 1);
    }
}

// Update all keyframes based on current turning points
function updateAllKeyframes(episode) {
    // Regenerate annotations from turning points
    const keyframes = episode.keyframes || [];
    currentAnnotations = generateAnnotationsFromTurningPoints(
        turningPoints,
        keyframes,
        episode.num_frames || 0,
        episode.num_subtasks || 1
    );
    
    // Update the display
    const grid = document.getElementById('keyframesGrid');
    if (grid) {
        // Update each keyframe item
        grid.querySelectorAll('.keyframe-item').forEach(item => {
            const frameIdx = parseInt(item.dataset.frameIdx);
            const inferredSubtask = getInferredSubtask(
                frameIdx,
                episode.keyframes || [],
                episode.num_frames || 0
            );
            
            // Update inferred subtask label
            const label = item.querySelector('.current-label');
            if (label) {
                label.textContent = `Inferred: ${inferredSubtask}`;
            }
            
            // Update turning point indicator
            if (isTurningPoint(frameIdx)) {
                item.classList.add('turning-point');
                const header = item.querySelector('.frame-header');
                if (header) {
                    let headerRight = header.querySelector('.header-right');
                    if (!headerRight) {
                        headerRight = document.createElement('div');
                        headerRight.className = 'header-right';
                        const oldLabel = header.querySelector('.current-label');
                        if (oldLabel) {
                            headerRight.appendChild(oldLabel);
                        }
                        header.appendChild(headerRight);
                    }
                    if (!headerRight.querySelector('.tp-badge')) {
                        const badge = document.createElement('span');
                        badge.className = 'tp-badge';
                        badge.textContent = '★ Turning Point';
                        headerRight.insertBefore(badge, headerRight.firstChild);
                    }
                }
            } else {
                item.classList.remove('turning-point');
                const badge = item.querySelector('.tp-badge');
                if (badge) {
                    badge.remove();
                }
            }
        });
    }
}

// Show subtask selector for turning point
function showSubtaskSelectorForTP(frameIdx, item, tpControl, currentSubtaskIdx = null) {
    // Remove existing selector if any
    const existingSelector = item.querySelector('.tp-subtask-selector');
    if (existingSelector) {
        existingSelector.remove();
    }
    
    const selector = document.createElement('div');
    selector.className = 'tp-subtask-selector';
    
    const label = document.createElement('label');
    label.textContent = 'Subtask starts here:';
    selector.appendChild(label);
    
    const buttons = document.createElement('div');
    buttons.className = 'choice-buttons';
    
    planChoices.forEach((choice, index) => {
        const subtaskIndex = index + 1;
        const button = document.createElement('button');
        button.className = 'choice-button';
        button.type = 'button';
        if (currentSubtaskIdx === subtaskIndex || (!currentSubtaskIdx && subtaskIndex === 1)) {
            button.classList.add('selected');
            if (!currentSubtaskIdx) {
                currentSubtaskIdx = subtaskIndex;
            }
        }
        button.textContent = `${String.fromCharCode(65 + index)}: ${choice}`;
        button.dataset.subtaskIndex = subtaskIndex;
        
        button.addEventListener('click', function() {
            buttons.querySelectorAll('.choice-button').forEach(btn => {
                btn.classList.remove('selected');
            });
            this.classList.add('selected');
            
            // Add/update turning point
            addOrUpdateTurningPoint(frameIdx, subtaskIndex);
            updateAllKeyframes(currentEpisode);
        });
        
        buttons.appendChild(button);
    });
    
    selector.appendChild(buttons);
    tpControl.appendChild(selector);
    
    // If no current subtask, set default to first
    if (!currentSubtaskIdx) {
        addOrUpdateTurningPoint(frameIdx, 1);
        updateAllKeyframes(currentEpisode);
    }
}

// Create a keyframe item element
function createKeyframeItem(frameIdx, imageData, episode) {
    const item = document.createElement('div');
    item.className = 'keyframe-item';
    item.dataset.frameIdx = frameIdx;
    
    // Check if this is a turning point
    const isTP = isTurningPoint(frameIdx);
    if (isTP) {
        item.classList.add('turning-point');
    }
    
    // Get inferred subtask from turning points
    const inferredSubtask = getInferredSubtask(
        frameIdx, 
        episode.keyframes || [], 
        episode.num_frames || 0
    );
    
    // Find current annotation (for reference)
    const annotation = currentAnnotations.find(a => a.frame_index === frameIdx);
    const originalSubtask = annotation ? annotation.subtask_index : null;
    
    // Frame header
    const header = document.createElement('div');
    header.className = 'frame-header';
    
    const frameIndexSpan = document.createElement('span');
    frameIndexSpan.className = 'frame-index';
    frameIndexSpan.textContent = `Frame ${frameIdx}`;
    header.appendChild(frameIndexSpan);
    
    const headerRight = document.createElement('div');
    headerRight.className = 'header-right';
    
    if (isTP) {
        const badge = document.createElement('span');
        badge.className = 'tp-badge';
        badge.textContent = '★ Turning Point';
        headerRight.appendChild(badge);
    }
    
    const label = document.createElement('span');
    label.className = 'current-label';
    label.textContent = `Inferred: ${inferredSubtask}`;
    headerRight.appendChild(label);
    
    header.appendChild(headerRight);
    item.appendChild(header);
    
    // Frame image
    const imageContainer = document.createElement('div');
    imageContainer.className = 'frame-image';
    if (imageData) {
        const img = document.createElement('img');
        img.src = imageData;
        img.alt = `Frame ${frameIdx}`;
        imageContainer.appendChild(img);
    } else {
        imageContainer.textContent = 'Image not available';
    }
    item.appendChild(imageContainer);
    
    // Turning point control
    const tpControl = document.createElement('div');
    tpControl.className = 'turning-point-control';
    
    const tpCheckbox = document.createElement('input');
    tpCheckbox.type = 'checkbox';
    tpCheckbox.id = `tp-${frameIdx}`;
    tpCheckbox.checked = isTP;
    tpCheckbox.addEventListener('change', function() {
        if (this.checked) {
            // Mark as turning point - need to select subtask
            showSubtaskSelectorForTP(frameIdx, item, tpControl);
            item.classList.add('turning-point');
        } else {
            // Remove turning point
            removeTurningPoint(frameIdx);
            item.classList.remove('turning-point');
            const badge = item.querySelector('.tp-badge');
            if (badge) {
                badge.remove();
            }
            const selector = item.querySelector('.tp-subtask-selector');
            if (selector) {
                selector.remove();
            }
            updateAllKeyframes(episode);
        }
    });
    
    const tpLabel = document.createElement('label');
    tpLabel.htmlFor = `tp-${frameIdx}`;
    tpLabel.textContent = 'Mark as Turning Point';
    tpLabel.className = 'tp-label';
    
    tpControl.appendChild(tpCheckbox);
    tpControl.appendChild(tpLabel);
    
    // Show subtask selector if it's a turning point
    if (isTP) {
        const tp = turningPoints.find(t => t.frame_index === frameIdx);
        if (tp) {
            showSubtaskSelectorForTP(frameIdx, item, tpControl, tp.subtask_index);
        }
    }
    
    item.appendChild(tpControl);
    
    // Reference annotation (read-only display) if different from inferred
    if (originalSubtask && originalSubtask !== inferredSubtask) {
        const refDiv = document.createElement('div');
        refDiv.className = 'reference-annotation';
        refDiv.textContent = `Original: ${originalSubtask}`;
        item.appendChild(refDiv);
    }
    
    return item;
}

// Update annotation for a frame (legacy function, kept for compatibility)
function updateAnnotation(frameIdx, subtaskIndex) {
    const existingIndex = currentAnnotations.findIndex(a => a.frame_index === frameIdx);
    if (existingIndex >= 0) {
        currentAnnotations[existingIndex].subtask_index = subtaskIndex;
    } else {
        currentAnnotations.push({
            frame_index: frameIdx,
            subtask_index: subtaskIndex
        });
    }
    
    // Sort annotations by frame_index
    currentAnnotations.sort((a, b) => a.frame_index - b.frame_index);
}

// Save changes
async function saveChanges() {
    const saveBtn = document.getElementById('saveBtn');
    const saveStatus = document.getElementById('saveStatus');
    
    saveBtn.disabled = true;
    saveStatus.textContent = 'Saving...';
    saveStatus.className = 'save-status';
    
    try {
        // Ensure annotations are up to date from turning points
        if (turningPoints.length > 0 && currentEpisode) {
            const keyframes = currentEpisode.keyframes || [];
            currentAnnotations = generateAnnotationsFromTurningPoints(
                turningPoints,
                keyframes,
                currentEpisode.num_frames || 0,
                currentEpisode.num_subtasks || 1
            );
        }
        
        const response = await fetch('/api/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                episode_key: currentEpisode.episode_key,
                annotations: currentAnnotations
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to save');
        }
        
        const data = await response.json();
        
        // Update original annotations
        originalAnnotations = JSON.parse(JSON.stringify(currentAnnotations));
        
        // Remove modified markers
        document.querySelectorAll('.keyframe-item.modified').forEach(item => {
            item.classList.remove('modified');
        });
        
        saveStatus.textContent = 'Saved successfully!';
        saveStatus.className = 'save-status success';
        
        // Update episode data
        currentEpisode.subtask_sequence = data.subtask_sequence;
        currentEpisode.subtask_counts = data.subtask_counts;
        
    } catch (error) {
        console.error('Error saving:', error);
        saveStatus.textContent = 'Error: ' + error.message;
        saveStatus.className = 'save-status error';
    } finally {
        saveBtn.disabled = false;
    }
}

// Hide episode data
function hideEpisodeData() {
    document.getElementById('episodeInfo').style.display = 'none';
    document.getElementById('planSection').style.display = 'none';
    document.getElementById('keyframesContainer').style.display = 'none';
    document.getElementById('saveSection').style.display = 'none';
    currentEpisode = null;
    currentEpisodeIndex = -1;
    currentAnnotations = [];
    originalAnnotations = [];
    planChoices = [];
    turningPoints = [];
    updateNavigationButtons();
}

// Show error message
function showError(message) {
    const container = document.querySelector('.container');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    container.insertBefore(errorDiv, container.firstChild);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

