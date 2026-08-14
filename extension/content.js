const LOCAL_API = 'http://127.0.0.1:8000/api/sync';
const DEFAULT_MAX_SUBMISSIONS = 1000;
const PAGE_SIZE = 20;

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const getCsrfToken = () => {
  const cookies = document.cookie.split(';');
  for (const c of cookies) {
    const [name, val] = c.trim().split('=');
    if (name && name.toLowerCase().includes('csrftoken')) {
      return decodeURIComponent(val);
    }
  }
  const meta = document.querySelector('meta[name="csrf-token"], meta[name="csrf-param"]');
  if (meta && meta.content) return meta.content;
  return '';
};

const ensureCsrfToken = async () => {
  let token = getCsrfToken();
  if (token) return token;

  try {
    await fetch('/', { method: 'GET', credentials: 'include' });
    token = getCsrfToken();
  } catch (e) {
    console.warn('Could not fetch homepage for CSRF token:', e);
  }

  return token;
};

const buildAuthHeaders = async () => {
  const headers = {
    'Accept': 'application/json, text/plain, */*',
    'X-Requested-With': 'XMLHttpRequest',
  };
  const csrf = await ensureCsrfToken();
  if (csrf) {
    headers['x-csrftoken'] = csrf;
    headers['X-CSRFToken'] = csrf;
  }
  return headers;
};

const parseFloatValue = (value) => {
  if (value == null || value === '' || value === 'N/A') return null;
  if (typeof value === 'number') return value;
  const match = String(value).match(/-?\d+(?:\.\d+)?/);
  return match ? Number(match[0]) : null;
};

const normalizeString = (value) => {
  if (value == null || value === '') return null;
  return String(value);
};

const normalizeTopics = (rawTopics) => {
  if (!rawTopics) return [];
  if (!Array.isArray(rawTopics)) {
    if (typeof rawTopics === 'string') return [rawTopics];
    return [];
  }
  return rawTopics
    .map((t) => {
      if (!t) return null;
      if (typeof t === 'string') return t;
      if (typeof t === 'object') return t.name || t.slug || t.title || null;
      return String(t);
    })
    .filter(Boolean);
};

const normalizeLeetCodeResponse = (username, rawItems) => {
  const seenKeys = new Set();
  const normalized = [];

  for (const item of rawItems) {
    const subId = normalizeString(item.id || item.submission_id || item.submit_id);
    const slug = normalizeString(item.title_slug || item.titleSlug || item.question_slug || item.problem_slug || item.question_id);
    const title = normalizeString(item.title || item.problem_title || item.title_slug || item.titleSlug || item.question_id);
    const timestamp = normalizeString(item.timestamp || item.time || item.submission_date);

    if (!subId || !slug) continue;

    const key = subId || `${slug}_${timestamp || ''}`;
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);

    normalized.push({
      submission_id: subId,
      problem_slug: slug,
      problem_title: title || slug,
      difficulty: normalizeString(item.difficulty || item.question_difficulty) || 'Medium',
      status: normalizeString(item.status_display || item.statusDisplay || item.status || item.result) || 'Accepted',
      language: normalizeString(item.lang || item.language) || 'Python3',
      user_id: 1,
      timestamp: timestamp || String(Math.floor(Date.now() / 1000)),
      runtime: parseFloatValue(item.runtime || item.runtime_ms),
      memory: parseFloatValue(item.memory),
      topics: normalizeTopics(item.topic_tags || item.topics || item.tags),
    });
  }

  return normalized;
};

const fetchUsernameFromGraphQL = async () => {
  try {
    const query = `
      query globalData {
        userStatus {
          username
          isSignedIn
        }
      }
    `;
    const headers = {
      'Content-Type': 'application/json',
      ...(await buildAuthHeaders()),
    };
    const resp = await fetch('/graphql', {
      method: 'POST',
      headers,
      credentials: 'include',
      body: JSON.stringify({ query }),
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data?.data?.userStatus?.username) {
        return data.data.userStatus.username;
      }
    }
  } catch (err) {
    console.warn('Unable to fetch username via GraphQL:', err);
  }
  return null;
};

const extractUsernameFromPage = async () => {
  // 1. Check URL path
  const urlMatch = window.location.pathname.match(/\/(?:u|profile)\/([^\/]+)/);
  if (urlMatch && urlMatch[1]) {
    return decodeURIComponent(urlMatch[1]).trim();
  }

  // 2. Check GraphQL
  const gqlUser = await fetchUsernameFromGraphQL();
  if (gqlUser) return gqlUser;

  // 3. Check profile link in DOM
  const profileLink = document.querySelector('a[href^="/profile/"]') || document.querySelector('a[href^="/u/"]');
  if (profileLink) {
    const text = profileLink.textContent.trim();
    if (text) return text;
  }

  // 4. Fetch last active username from backend status endpoint if available
  try {
    const statusResp = await fetch('http://127.0.0.1:8000/api/status');
    if (statusResp.ok) {
      const data = await statusResp.json();
      if (data.last_username) return data.last_username;
    }
  } catch (e) {
    // Backend fetch fallback
  }

  return 'leetcode_user';
};

const fetchPageREST = async (offset, limit) => {
  const headers = await buildAuthHeaders();
  const resp = await fetch(`/api/submissions/?offset=${offset}&limit=${limit}`, {
    method: 'GET',
    headers,
    credentials: 'include',
  });
  if (!resp.ok) {
    throw new Error(`REST returned status ${resp.status}`);
  }
  const text = await resp.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error('REST response is not valid JSON');
  }
};

const fetchPageGraphQL = async (offset, limit) => {
  const headers = {
    'Content-Type': 'application/json',
    ...(await buildAuthHeaders()),
  };
  const query = `
    query submissionList($offset: Int!, $limit: Int!) {
      submissionList(offset: $offset, limit: $limit) {
        hasNext
        submissions {
          id
          title
          titleSlug
          statusDisplay
          lang
          timestamp
          url
          isPending
          memory
          runtime
        }
      }
    }
  `;
  const resp = await fetch('/graphql', {
    method: 'POST',
    headers,
    credentials: 'include',
    body: JSON.stringify({
      query,
      variables: { offset, limit },
    }),
  });
  if (!resp.ok) {
    throw new Error(`GraphQL returned status ${resp.status}`);
  }
  const data = await resp.json();
  if (data.errors && data.errors.length > 0) {
    throw new Error(`GraphQL error: ${data.errors[0].message}`);
  }
  return data;
};

const parseSubmissionData = (data, limit) => {
  if (!data) return null;

  let items = null;
  let hasNext = null;

  if (Array.isArray(data)) {
    items = data;
  } else if (Array.isArray(data.submissions)) {
    items = data.submissions;
    hasNext = data.has_next ?? data.hasNext ?? data.has_more;
  } else if (Array.isArray(data.submission_list)) {
    items = data.submission_list;
    hasNext = data.has_next ?? data.hasNext ?? data.has_more;
  } else if (Array.isArray(data.submissions_dump)) {
    items = data.submissions_dump;
    hasNext = data.has_next ?? data.hasNext ?? data.has_more;
  } else if (data?.data?.submissionList) {
    const list = data.data.submissionList;
    items = Array.isArray(list.submissions) ? list.submissions : [];
    hasNext = list.hasNext;
  } else if (data?.data?.recentSubmissions) {
    const recent = data.data.recentSubmissions;
    items = Array.isArray(recent)
      ? recent
      : Array.isArray(recent.edges)
      ? recent.edges.map((e) => e.node || e)
      : [];
    hasNext = false;
  }

  if (!items) return null;

  // Determine hasNext accurately so pagination continues beyond 20 items
  if (typeof hasNext !== 'boolean') {
    if (data.last_key != null && data.last_key !== '') {
      hasNext = true;
    } else {
      hasNext = items.length >= limit;
    }
  }

  return { items, hasNext };
};

const fetchSubmissionPage = async (offset, limit) => {
  let restError = null;

  // Primary: REST API with authentication headers
  try {
    const restData = await fetchPageREST(offset, limit);
    const parsed = parseSubmissionData(restData, limit);
    if (parsed) return parsed;
  } catch (err) {
    restError = err;
    console.warn(`REST submission fetch at offset ${offset} failed, attempting GraphQL fallback: ${err.message}`);
  }

  // Fallback: GraphQL API with authentication headers
  try {
    const gqlData = await fetchPageGraphQL(offset, limit);
    const parsed = parseSubmissionData(gqlData, limit);
    if (parsed) return parsed;
  } catch (gqlErr) {
    console.warn(`GraphQL submission fetch at offset ${offset} failed: ${gqlErr.message}`);
    throw new Error(`LeetCode submissions request failed: ${restError ? restError.message : gqlErr.message}`);
  }

  throw new Error('Unable to parse LeetCode submissions response from either REST or GraphQL endpoint.');
};

const syncSubmissions = async (maxItems = DEFAULT_MAX_SUBMISSIONS, syncMode = 'sync_new') => {
  const username = await extractUsernameFromPage();
  let allRawSubmissions = [];
  let offset = 0;
  const pageLimit = PAGE_SIZE;

  while (allRawSubmissions.length < maxItems) {
    const pageResult = await fetchSubmissionPage(offset, pageLimit);
    const { items, hasNext } = pageResult;

    if (!items || items.length === 0) break;

    allRawSubmissions.push(...items);

    if (hasNext === false || items.length < pageLimit) break;

    offset += items.length;

    if (allRawSubmissions.length < maxItems) {
      await delay(100);
    }
  }

  if (allRawSubmissions.length > maxItems) {
    allRawSubmissions = allRawSubmissions.slice(0, maxItems);
  }

  const normalized = normalizeLeetCodeResponse(username, allRawSubmissions);
  const finalSubmissions = normalized.slice(0, maxItems);
  if (finalSubmissions.length === 0) {
    throw new Error('No valid submissions found in LeetCode response.');
  }

  const isRebuild = syncMode === 'rebuild';
  const postPayload = {
    username,
    submissions: finalSubmissions,
    sync_mode: syncMode,
    rebuild_history: isRebuild,
    fetch_limit: maxItems
  };

  console.log('[LeetCode Mentor Content Script] POSTing to /api/sync:', {
    username,
    submission_count: finalSubmissions.length,
    sync_mode: syncMode,
    rebuild_history: isRebuild,
    fetch_limit: maxItems
  });

  const resp = await fetch(LOCAL_API, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(postPayload),
  });

  if (!resp.ok) {
    const errorText = await resp.text();
    throw new Error(`Local API sync failed: ${resp.status} ${errorText}`);
  }

  return resp.json();
};

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== 'START_SYNC') {
    return false;
  }

  const maxItems = parseInt(message?.maxItems || message?.fetchLimit || DEFAULT_MAX_SUBMISSIONS, 10);
  const isRebuild = message?.syncMode === 'rebuild' || Boolean(message?.rebuildHistory);
  const syncMode = isRebuild ? 'rebuild' : 'sync_new';

  console.log(`[LeetCode Mentor Content Script] Received START_SYNC: maxItems=${maxItems}, syncMode=${syncMode}, isRebuild=${isRebuild}`);

  syncSubmissions(maxItems, syncMode)
    .then((result) => sendResponse(result))
    .catch((error) => sendResponse({ error: error.message }));

  return true;
});



