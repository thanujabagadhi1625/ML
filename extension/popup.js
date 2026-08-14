const syncNewBtn = document.getElementById('syncNewBtn');
const rebuildBtn = document.getElementById('rebuildBtn');
const maxItemsSelect = document.getElementById('maxItems');
const storedCountEl = document.getElementById('storedCount');
const lastSyncEl = document.getElementById('lastSync');
const statusEl = document.getElementById('status');
const progressEl = document.getElementById('progress');
const detailsEl = document.getElementById('details');

const LOCAL_STATUS_API = 'http://127.0.0.1:8000/api/status';

const formatLastSync = (isoString) => {
  if (!isoString) return 'Never';
  try {
    const d = new Date(isoString);
    return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch (e) {
    return isoString;
  }
};

const updateStatusFromBackend = async () => {
  try {
    const res = await fetch(LOCAL_STATUS_API);
    if (res.ok) {
      const data = await res.json();
      const total = data.total_submissions || 0;
      storedCountEl.textContent = `${total} submission${total === 1 ? '' : 's'}`;
      lastSyncEl.textContent = `Last sync: ${formatLastSync(data.last_synced)}`;
      return total;
    }
  } catch (e) {
    storedCountEl.textContent = 'Offline (Server stopped)';
    lastSyncEl.textContent = 'Start FastAPI server (python files/api_server.py)';
  }
  return 0;
};

// Initial status fetch on popup open
updateStatusFromBackend();

const handleSync = async (syncMode) => {
  const isRebuild = syncMode === 'rebuild';
  const modeLabel = isRebuild ? 'Rebuilding History' : 'Syncing New';

  statusEl.textContent = `${modeLabel}...`;
  progressEl.textContent = 'Fetching submissions from LeetCode...';
  detailsEl.textContent = '';
  syncNewBtn.disabled = true;
  rebuildBtn.disabled = true;

  const maxItems = parseInt(maxItemsSelect?.value || '1000', 10);

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    statusEl.textContent = 'Unable to find active browser tab.';
    syncNewBtn.disabled = false;
    rebuildBtn.disabled = false;
    return;
  }

  if (!tab.url || !tab.url.includes('leetcode.com')) {
    statusEl.textContent = 'Please open a LeetCode page.';
    detailsEl.textContent = 'Navigate to https://leetcode.com and ensure you are logged in.';
    syncNewBtn.disabled = false;
    rebuildBtn.disabled = false;
    return;
  }

  console.log(`[LeetCode Mentor Popup] Initiating sync: mode=${syncMode}, maxItems=${maxItems}, isRebuild=${isRebuild}`);
  chrome.tabs.sendMessage(tab.id, { type: 'START_SYNC', maxItems, syncMode, rebuildHistory: isRebuild, fetchLimit: maxItems }, (response) => {
    syncNewBtn.disabled = false;
    rebuildBtn.disabled = false;

    if (chrome.runtime.lastError) {
      statusEl.textContent = 'Extension script not loaded on page.';
      detailsEl.textContent = chrome.runtime.lastError.message + '. Please refresh your LeetCode tab and try again.';
      return;
    }

    if (!response) {
      statusEl.textContent = 'No response from page context.';
      return;
    }

    if (response.error) {
      statusEl.textContent = 'Sync failed.';
      detailsEl.textContent = response.error;
      return;
    }

    statusEl.textContent = isRebuild ? 'Rebuild Complete!' : 'Sync Complete!';

    if (isRebuild) {
      progressEl.textContent = `History limit: ${maxItems} | Rebuilt stored history with ${response.total_submissions} submissions.`;
      detailsEl.textContent = `User '${response.username}' history replaced cleanly.`;
    } else {
      const added = response.new_submissions || 0;
      const found = response.new_found || 0;
      if (added > 0) {
        progressEl.textContent = `New found: ${found} | Added: ${added} | Total stored: ${response.total_submissions}`;
        detailsEl.textContent = `Successfully updated user '${response.username}'.`;
      } else {
        progressEl.textContent = `New found: 0 | Nothing new to add.`;
        detailsEl.textContent = `Total stored: ${response.total_submissions} submissions remain intact.`;
      }
    }

    updateStatusFromBackend();
  });
};

syncNewBtn.addEventListener('click', () => handleSync('sync_new'));
rebuildBtn.addEventListener('click', () => handleSync('rebuild'));
