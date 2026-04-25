import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

const viewerEl = document.getElementById('viewer');
const statusText = document.getElementById('status-text');
const promptEl = document.getElementById('prompt');
const goBtn = document.getElementById('go');
const previewImg = document.getElementById('preview');
const previewWrap = document.getElementById('preview-wrap');
const downloadBtn = document.getElementById('download');
const animSelect = document.getElementById('anim-select');
const tgSkel = document.getElementById('tg-skel');
const tgWire = document.getElementById('tg-wire');
const tgGrid = document.getElementById('tg-grid');
const tgAutorot = document.getElementById('tg-autorot');
const jobInfo = document.getElementById('jobinfo');
const modelSelect = document.getElementById('model-select');
const styleSelect = document.getElementById('style-select');
const enrichedText = document.getElementById('enriched-text');
const modelMeta = document.getElementById('model-meta');
const cancelBtn = document.getElementById('cancel');

let activeJobId = null;

let modelsInfo = null;

// === Settings ===============================================================
// Persisted per-user via localStorage; sent with every /api/generate.
// `defaults` is hydrated from /api/defaults on load so the server is the
// source of truth for what each field should start as.
let serverDefaults = null;
let userSettings = {};

const SETTINGS_KEY = 'anigen_settings_v1';

function loadUserSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
  } catch { return {}; }
}

function saveUserSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(userSettings));
}

async function initSettings() {
  userSettings = loadUserSettings();
  try {
    const r = await fetch('/api/defaults');
    serverDefaults = await r.json();
  } catch (e) {
    console.warn('defaults fetch failed', e);
    serverDefaults = {t2i:{}, anigen:{}, hunyuan3d:{}};
  }
  populateModalFields();
}

function field(section, key, fallback) {
  if (userSettings[section]?.[key] !== undefined && userSettings[section]?.[key] !== '')
    return userSettings[section][key];
  return (serverDefaults?.[section]?.[key]) ?? fallback;
}

function populateModalFields() {
  const set = (id, v) => { const el = document.getElementById(id); if (el) {
    if (el.type === 'checkbox') el.checked = !!v;
    else el.value = v ?? '';
  }};
  set('s-t2i-suffix', field('t2i', 'custom_suffix', ''));
  set('s-t2i-negative', field('t2i', 'custom_negative', ''));
  set('s-t2i-steps', field('t2i', 'num_inference_steps', 35));
  set('s-t2i-guidance', field('t2i', 'guidance_scale', 7.5));
  set('s-t2i-pag', field('t2i', 'pag_scale', 2.0));

  set('s-anigen-cfg-ss', field('anigen', 'cfg_scale_ss', 7.5));
  set('s-anigen-cfg-slat', field('anigen', 'cfg_scale_slat', 3.0));
  set('s-anigen-joints', field('anigen', 'joints_density', 1));
  set('s-anigen-smooth-iters', field('anigen', 'smooth_skin_weights_iters', 100));
  set('s-anigen-smooth-alpha', field('anigen', 'smooth_skin_weights_alpha', 1.0));
  set('s-anigen-smooth', field('anigen', 'smooth_skin_weights', true));
  set('s-anigen-filter', field('anigen', 'filter_skin_weights', true));

  set('s-hy3d-texture', field('hunyuan3d', 'enable_texture', true));
  set('s-hy3d-bg', field('hunyuan3d', 'enable_bg_removal', true));
  set('s-hy3d-polys', field('hunyuan3d', 'max_polygons', 120000));
}

function collectModalFields() {
  const get = (id) => document.getElementById(id);
  const n = (id) => {
    const v = get(id)?.value;
    if (v === '' || v === undefined) return undefined;
    const f = parseFloat(v);
    return Number.isFinite(f) ? f : undefined;
  };
  const b = (id) => !!get(id)?.checked;
  const s = (id) => (get(id)?.value || '').trim();
  userSettings = {
    t2i: {
      custom_suffix: s('s-t2i-suffix'),
      custom_negative: s('s-t2i-negative'),
      num_inference_steps: n('s-t2i-steps'),
      guidance_scale: n('s-t2i-guidance'),
      pag_scale: n('s-t2i-pag'),
    },
    anigen: {
      cfg_scale_ss: n('s-anigen-cfg-ss'),
      cfg_scale_slat: n('s-anigen-cfg-slat'),
      joints_density: n('s-anigen-joints'),
      smooth_skin_weights_iters: n('s-anigen-smooth-iters'),
      smooth_skin_weights_alpha: n('s-anigen-smooth-alpha'),
      smooth_skin_weights: b('s-anigen-smooth'),
      filter_skin_weights: b('s-anigen-filter'),
    },
    hunyuan3d: {
      enable_texture: b('s-hy3d-texture'),
      enable_bg_removal: b('s-hy3d-bg'),
      max_polygons: n('s-hy3d-polys'),
    },
  };
}

function openSettings() {
  populateModalFields();
  document.getElementById('settings-modal').style.display = 'flex';
  activateTab('t2i');
}

function closeSettings() {
  document.getElementById('settings-modal').style.display = 'none';
}

function activateTab(tab) {
  document.querySelectorAll('#settings-tabs .tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-pane').forEach(p =>
    p.style.display = p.dataset.tab === tab ? 'block' : 'none');
}

document.getElementById('settings-btn').addEventListener('click', openSettings);
document.getElementById('modal-close').addEventListener('click', closeSettings);
document.getElementById('modal-cancel').addEventListener('click', closeSettings);
document.getElementById('modal-save').addEventListener('click', () => {
  collectModalFields();
  saveUserSettings();
  closeSettings();
});
document.getElementById('modal-reset').addEventListener('click', () => {
  userSettings = {};
  localStorage.removeItem(SETTINGS_KEY);
  populateModalFields();
});
document.querySelectorAll('#settings-tabs .tab-btn').forEach(b =>
  b.addEventListener('click', () => activateTab(b.dataset.tab)));
document.getElementById('settings-modal').addEventListener('click', (e) => {
  if (e.target.id === 'settings-modal') closeSettings();
});

initSettings();

async function loadHealthInfo() {
  try {
    const h = await fetch('/api/health').then(r => r.json());
    modelsInfo = h.models;
    // Update model select with readiness
    [...modelSelect.options].forEach(opt => {
      const info = modelsInfo[opt.value];
      if (info) {
        let label = info.label;
        if (!info.ready) label += ' (unavailable)';
        opt.textContent = label;
        opt.disabled = !info.ready;
      }
    });
    // If current selection is unavailable, switch to first available
    if (modelsInfo[modelSelect.value] && !modelsInfo[modelSelect.value].ready) {
      for (const opt of modelSelect.options) {
        if (!opt.disabled) { modelSelect.value = opt.value; break; }
      }
    }
    updateModelMeta();
  } catch (e) { console.warn('health fetch failed', e); }
}

function updateModelMeta() {
  if (!modelsInfo) { modelMeta.textContent = ''; return; }
  const cur = modelsInfo[modelSelect.value];
  if (!cur) { modelMeta.textContent = ''; return; }
  const badges = [];
  if (cur.rigged) badges.push('<span class="badge rigged">RIGGED</span>');
  else badges.push('<span class="badge static">STATIC</span>');
  if (cur.textured === false) badges.push('<span class="badge warn">UNTEXTURED</span>');
  else if (cur.textured || cur.rigged) badges.push('<span class="badge textured">TEXTURED</span>');
  if (cur.multiview) badges.push('<span class="badge rigged" style="background:#5a3a8e;">MULTI-VIEW</span>');
  let desc = '';
  if (modelSelect.value === 'anigen') {
    desc = 'Rigged mesh with bones. Procedural animations available.';
  } else if (modelSelect.value === 'hunyuan3d') {
    desc = cur.textured
      ? 'Textured static mesh from a single image.'
      : 'Untextured static mesh.';
  } else if (modelSelect.value === 'hunyuan3dmv') {
    desc = 'Multi-view shape reconstruction. Upload up to 4 views (front required) for stronger geometry.';
  }
  if (cur.error) desc += ` Error: ${cur.error}`;
  modelMeta.innerHTML = badges.join(' ') + '<br>' + desc;

  // Show/hide multi-view panel
  const mvPanel = document.getElementById('mv-panel');
  if (mvPanel) mvPanel.style.display = (modelSelect.value === 'hunyuan3dmv') ? 'block' : 'none';
}

let enrichTimer = null;
async function updateEnrichedPreview() {
  const p = promptEl.value.trim();
  if (!p) { enrichedText.textContent = '—'; return; }
  try {
    const r = await fetch('/api/enrich', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: p, model: modelSelect.value, style: styleSelect.value, settings: userSettings})
    });
    const d = await r.json();
    enrichedText.textContent = d.enriched || '—';
  } catch (e) { enrichedText.textContent = '(preview unavailable)'; }
}

function scheduleEnrichUpdate() {
  clearTimeout(enrichTimer);
  enrichTimer = setTimeout(updateEnrichedPreview, 250);
}

modelSelect.addEventListener('change', () => { updateModelMeta(); scheduleEnrichUpdate(); mirrorUploadToMvFront(); updateGenViewsBtn(); });
styleSelect.addEventListener('change', scheduleEnrichUpdate);
promptEl.addEventListener('input', scheduleEnrichUpdate);

loadHealthInfo();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a21);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.6, 1.2, 2.4);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
viewerEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.5, 0);

// Transform gizmo for manual rig editing. Attached to a bone when the user
// clicks a classification sphere; dragging moves the bone and the skinned
// mesh deforms live via its skinning weights.
const transformControls = new TransformControls(camera, renderer.domElement);
transformControls.setMode('translate');
transformControls.setSize(0.8);
transformControls.addEventListener('dragging-changed', (e) => {
  controls.enabled = !e.value;
});
// three.js r160+ exposes the gizmo via getHelper(); older versions ARE the
// Object3D. Handle both.
const transformGizmo = (typeof transformControls.getHelper === 'function')
  ? transformControls.getHelper()
  : transformControls;
scene.add(transformGizmo);
let editingBone = null;
const raycaster = new THREE.Raycaster();
const mouseNDC = new THREE.Vector2();

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);
const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(3, 5, 3);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0x8eb2ff, 0.8);
fillLight.position.set(-3, 2, -2);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
rimLight.position.set(0, 3, -5);
scene.add(rimLight);

const ENV_PRESETS = {
  studio: { ambient: 1.0, key: 4.0, fill: 1.8, rim: 1.2, exposure: 1.4, keyColor: 0xffffff, fillColor: 0x8eb2ff, rimColor: 0xffffff, bg: 0x1a1a21 },
  neutral: { ambient: 1.8, key: 3.0, fill: 1.8, rim: 0.8, exposure: 1.5, keyColor: 0xffffff, fillColor: 0xffffff, rimColor: 0xffffff, bg: 0x2a2a30 },
  sunset: { ambient: 0.6, key: 5.5, fill: 1.2, rim: 2.0, exposure: 1.3, keyColor: 0xffb47a, fillColor: 0x6090d0, rimColor: 0xff7040, bg: 0x1a0f15 },
  neon: { ambient: 0.5, key: 3.5, fill: 3.0, rim: 3.5, exposure: 1.2, keyColor: 0xff3399, fillColor: 0x33ccff, rimColor: 0x99ff66, bg: 0x090814 },
  night: { ambient: 0.3, key: 2.0, fill: 0.8, rim: 1.5, exposure: 1.1, keyColor: 0x8ea8ff, fillColor: 0x4060a0, rimColor: 0xd0e0ff, bg: 0x05070d },
};

let grid = new THREE.GridHelper(4, 16, 0x444450, 0x2a2a33);
grid.position.y = 0;
scene.add(grid);

let currentModel = null;
let skeletonHelper = null;
let classificationHelpers = null;  // colored spheres marking leg/arm/spine bones
let mixer = null;
let currentAction = null;
const clock = new THREE.Clock();

const CLASS_COLORS = {
  leg: 0xff4040,
  arm: 0x4080ff,
  spine: 0x40ff40,
  other: 0xffff40,
  root: 0xffffff,
};

function buildClassificationHelpers(skel, scale) {
  if (!skel) return null;
  const group = new THREE.Group();
  group.name = '__classification';
  const r = scale * 0.010;
  const geom = new THREE.SphereGeometry(r, 12, 12);
  const made = new Map();
  const add = (bone, type) => {
    if (made.has(bone)) return;
    // depthTest ON so the gizmo renders on top of the spheres when it overlaps.
    const m = new THREE.MeshBasicMaterial({ color: CLASS_COLORS[type] });
    const s = new THREE.Mesh(geom, m);
    s.userData.bone = bone;
    made.set(bone, s);
    group.add(s);
  };
  add(skel.rootBone, 'root');
  skel.legs.forEach(leg => leg.forEach(b => add(b, 'leg')));
  skel.arms.forEach(arm => arm.forEach(b => add(b, 'arm')));
  (skel.spine || []).forEach(b => add(b, 'spine'));
  skel.bones.forEach(b => { if (!made.has(b)) add(b, 'other'); });
  scene.add(group);
  return group;
}

function resize() {
  const w = viewerEl.clientWidth;
  const h = viewerEl.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  if (mixer) mixer.update(clock.getDelta());
  if (classificationHelpers && classificationHelpers.visible) {
    classificationHelpers.children.forEach(s => {
      s.userData.bone.getWorldPosition(s.position);
    });
  }
  controls.autoRotate = tgAutorot.checked;
  controls.autoRotateSpeed = 1.5;
  controls.update();
  renderer.render(scene, camera);
}
animate();

function clearModel() {
  detachEditor();
  if (currentModel) {
    scene.remove(currentModel);
    currentModel.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(m => m.dispose());
      }
    });
    currentModel = null;
  }
  if (skeletonHelper) {
    scene.remove(skeletonHelper);
    skeletonHelper = null;
  }
  if (classificationHelpers) {
    scene.remove(classificationHelpers);
    classificationHelpers.children.forEach(c => { c.geometry.dispose(); c.material.dispose(); });
    classificationHelpers = null;
  }
  mixer = null;
  currentAction = null;
  animSelect.innerHTML = '<option>— no clips —</option>';
  animSelect.disabled = true;
}

// Size of the current model — kept so camera preset buttons can reframe without
// re-traversing the scene graph.
let currentModelSize = new THREE.Vector3(1, 1, 1);

function frameObject(object) {
  // First, center the object using the (precise) mesh bbox.
  const box = new THREE.Box3().setFromObject(object, true);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  object.position.sub(center);
  object.position.y += size.y / 2;
  currentModelSize.copy(size);

  // If the skeleton has detectable legs, re-ground so the lowest ankle sits
  // on the grid plane rather than the mesh bbox bottom. Mesh bbox can include
  // hair, capes, props, or straight-down geometry that isn't the feet.
  object.updateMatrixWorld(true);
  try {
    const skel = analyzeSkeleton(object);
    if (skel && skel.legs.length > 0) {
      let minAnkleY = Infinity;
      skel.legs.forEach(leg => {
        const ankle = leg[leg.length - 1];
        const p = new THREE.Vector3();
        ankle.getWorldPosition(p);
        if (p.y < minAnkleY) minAnkleY = p.y;
      });
      if (Number.isFinite(minAnkleY)) {
        object.position.y -= minAnkleY;
        console.log('[ground] shifted by', (-minAnkleY).toFixed(3), 'so ankles sit at y=0');
      }
    }
  } catch (e) { console.warn('[ground] ankle detection failed', e); }

  setCameraView('34');
}

function setCameraView(view) {
  const h = currentModelSize.y || 1;
  // For humanoid characters the T/A-pose makes bbox much wider than tall.
  // Use the largest horizontal dimension (height OR width) to size the camera
  // distance so arms don't clip out of frame.
  const horiz = Math.max(currentModelSize.x, currentModelSize.z);
  const reference = Math.max(h, horiz);
  const dist = reference * 1.8;
  const targetY = h * 0.5;
  const views = {
    front: [0, targetY, dist],
    back:  [0, targetY, -dist],
    side:  [dist, targetY, 0],
    '34':  [dist * 0.7, targetY + h * 0.1, dist * 0.9],
    top:   [0, dist * 1.2, 0.01],
    fit:   [dist * 0.7, targetY + h * 0.1, dist * 0.9],
  };
  const [x, y, z] = views[view] || views['34'];
  camera.position.set(x, y, z);
  controls.target.set(0, targetY, 0);
  controls.update();
  document.querySelectorAll('.cam-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.view === view);
  });
}

document.querySelectorAll('.cam-btn').forEach(btn => {
  btn.addEventListener('click', () => setCameraView(btn.dataset.view));
});

function loadGLB(url) {
  clearModel();
  console.log('[viewer] loadGLB start', url);
  setStatus('Downloading mesh…', true);
  const loader = new GLTFLoader();
  const t0 = performance.now();
  loader.load(url, (gltf) => {
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    console.log('[viewer] GLB parsed in', dt, 's; scenes=', gltf.scenes.length);
    setStatus(`Mesh parsed in ${dt}s, rendering…`, true);
    currentModel = gltf.scene;
    let meshCount = 0, triCount = 0;
    currentModel.traverse(o => {
      if (o.isMesh && o.material) {
        meshCount++;
        if (o.geometry && o.geometry.index) triCount += o.geometry.index.count / 3;
        else if (o.geometry) triCount += o.geometry.attributes.position.count / 3;
        o.material.side = THREE.DoubleSide;
      }
    });
    console.log('[viewer] meshes=', meshCount, 'triangles=', Math.round(triCount));
    scene.add(currentModel);
    frameObject(currentModel);
    setTimeout(() => setStatus(`Mesh loaded: ${Math.round(triCount).toLocaleString()} triangles across ${meshCount} mesh(es).`), 50);

    let skel = null;
    currentModel.traverse(o => { if (o.isSkinnedMesh && !skel) skel = o.skeleton; });
    if (skel) {
      skeletonHelper = new THREE.SkeletonHelper(currentModel);
      skeletonHelper.visible = tgSkel.checked;
      scene.add(skeletonHelper);
    }

    // Build classified-bone visualization (red=leg, blue=arm, green=spine).
    try {
      const analyzed = analyzeSkeleton(currentModel);
      if (analyzed) {
        const scale = Math.max(currentModelSize.x, currentModelSize.y, currentModelSize.z) || 1;
        classificationHelpers = buildClassificationHelpers(analyzed, scale);
        classificationHelpers.visible = tgClass ? tgClass.checked : false;
      }
    } catch (e) { console.warn('classification viz failed', e); }

    const bakedClips = (gltf.animations || []).slice();
    const proceduralClips = buildProceduralClips(currentModel);
    const allClips = bakedClips.concat(proceduralClips);

    if (allClips.length > 0) {
      mixer = new THREE.AnimationMixer(currentModel);
      animSelect.innerHTML = '';
      allClips.forEach((clip, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        const isBaked = i < bakedClips.length;
        opt.textContent = isBaked ? (clip.name || `clip ${i}`) : clip.name;
        animSelect.appendChild(opt);
      });
      const restOpt = document.createElement('option');
      restOpt.value = -1;
      restOpt.textContent = '(bind pose)';
      animSelect.appendChild(restOpt);
      animSelect.disabled = false;
      playClip(allClips, 0);
      animSelect.onchange = () => {
        const idx = parseInt(animSelect.value, 10);
        if (idx === -1) {
          if (currentAction) currentAction.stop();
          currentAction = null;
        } else {
          playClip(allClips, idx);
        }
      };
    }
    applyWireframe();
  }, (progress) => {
    if (progress.lengthComputable) {
      const pct = ((progress.loaded / progress.total) * 100).toFixed(0);
      const mb = (progress.loaded / 1e6).toFixed(1);
      const tot = (progress.total / 1e6).toFixed(1);
      setStatus(`Downloading mesh: ${mb} / ${tot} MB (${pct}%)`, true);
    } else if (progress.loaded) {
      const mb = (progress.loaded / 1e6).toFixed(1);
      setStatus(`Downloading mesh: ${mb} MB…`, true);
    }
  }, (err) => {
    console.error('[viewer] GLB load error', err);
    setStatus(`Error loading model: ${err.message || err}`);
  });
}

function playClip(clips, i) {
  if (!mixer) return;
  if (currentAction) currentAction.stop();
  currentAction = mixer.clipAction(clips[i]);
  currentAction.reset();
  currentAction.setLoop(THREE.LoopRepeat, Infinity);
  currentAction.play();
}

// === Skeleton analysis =======================================================
// AniGen outputs rigged meshes whose bone topology varies per character. We
// classify bones into legs, arms, spine by walking the tree from the root and
// looking at world-space rest positions. This lets us build walk/wave/sit
// animations that actually target the right limbs.

function analyzeSkeleton(root3d) {
  const bones = [];
  root3d.traverse(o => { if (o.isBone) bones.push(o); });
  if (bones.length === 0) return null;

  const rootBone = bones.find(b => !b.parent || !b.parent.isBone) || bones[0];
  root3d.updateMatrixWorld(true);

  const worldPos = new Map();
  bones.forEach(b => {
    const p = new THREE.Vector3();
    b.getWorldPosition(p);
    worldPos.set(b, p);
  });
  const rootPos = worldPos.get(rootBone);

  // Pick the next child whose direction most closely matches `wantDir`
  // (a world-space direction vector). Stops when no child is well-aligned.
  function followDirection(fromBone, wantDir, minDot = 0.0) {
    let cur = fromBone;
    const chain = [cur];
    while (true) {
      const kids = cur.children.filter(c => c.isBone);
      if (kids.length === 0) break;
      const curPos = worldPos.get(cur);
      let best = null, bestScore = -Infinity;
      for (const k of kids) {
        const d = worldPos.get(k).clone().sub(curPos);
        const len = d.length();
        if (len < 1e-5) continue;
        d.divideScalar(len);
        const score = d.dot(wantDir);
        if (score > bestScore) { bestScore = score; best = k; }
      }
      if (!best || bestScore < minDot) break;
      cur = best;
      chain.push(cur);
    }
    return chain;
  }

  function subtreeDepth(bone) {
    const kids = bone.children.filter(c => c.isBone);
    if (kids.length === 0) return 0;
    return 1 + Math.max(...kids.map(subtreeDepth));
  }
  function longestChain(fromBone) {
    let cur = fromBone;
    const chain = [cur];
    while (true) {
      const kids = cur.children.filter(c => c.isBone);
      if (kids.length === 0) break;
      let best = null, bestD = -1;
      for (const c of kids) {
        const d = subtreeDepth(c);
        if (d > bestD) { bestD = d; best = c; }
      }
      cur = best;
      chain.push(cur);
    }
    return chain;
  }

  const DOWN = new THREE.Vector3(0, -1, 0);
  const UP = new THREE.Vector3(0, 1, 0);

  const rootKids = rootBone.children.filter(c => c.isBone);
  const legs = [];
  let spine = null;
  const other = [];

  // Classify each root child by probing it in both directions and using the
  // LEAF position of the probe to decide. AniGen hip bones sit nearly at the
  // same Y as the root — only the leaf of the leg chain is clearly below.
  rootKids.forEach(child => {
    const downChain = followDirection(child, DOWN, 0.2);
    const downLeaf = worldPos.get(downChain[downChain.length - 1]);
    const downDy = downLeaf.y - rootPos.y;

    const upChain = followDirection(child, UP, 0.3);
    const upLeaf = worldPos.get(upChain[upChain.length - 1]);
    const upDy = upLeaf.y - rootPos.y;

    if (downChain.length >= 2 && downDy < -0.05) {
      legs.push(downChain);
    } else if (upChain.length >= 2 && upDy > 0.05) {
      if (!spine || upChain.length > spine.length) {
        if (spine) other.push(spine);
        spine = upChain;
      } else {
        other.push(upChain);
      }
    } else {
      other.push(longestChain(child));
    }
  });

  // Arms: any non-spine bone-chain branching off the spine.
  const arms = [...other];
  if (spine) {
    spine.forEach(sb => {
      sb.children.filter(c => c.isBone && !spine.includes(c)).forEach(armStart => {
        arms.push(longestChain(armStart));
      });
    });
  }

  legs.sort((a, b) => worldPos.get(a[0]).x - worldPos.get(b[0]).x);
  arms.sort((a, b) => worldPos.get(a[0]).x - worldPos.get(b[0]).x);

  console.log('[skel]', {
    bones: bones.length,
    legs: legs.map(l => l.map(b => b.name)),
    arms: arms.map(a => a.map(b => b.name)),
    spine: spine && spine.map(b => b.name),
  });

  return { bones, rootBone, worldPos, legs, arms, spine, rootPos };
}

// === Track helpers ===========================================================
// Build a QuaternionKeyframeTrack that rotates a bone around a *world-space*
// axis. We convert to the parent's local frame so the rotation behaves as
// intended regardless of the bone's rest orientation.

function makeTimes(duration, steps) {
  const t = new Float32Array(steps + 1);
  for (let i = 0; i <= steps; i++) t[i] = (i / steps) * duration;
  return t;
}

function localAxisFromWorld(bone, worldAxis) {
  const axis = worldAxis.clone();
  if (bone.parent) {
    const parentQ = new THREE.Quaternion();
    bone.parent.getWorldQuaternion(parentQ);
    axis.applyQuaternion(parentQ.invert());
  }
  return axis.normalize();
}

// Compute the bone's "swing axis" — perpendicular to the bone→child direction
// and world up. Rotating a leg around this axis swings it forward-back
// regardless of how the character is oriented in world space.
function boneSwingAxis(bone) {
  const childBones = bone.children.filter(c => c.isBone);
  if (childBones.length === 0) return new THREE.Vector3(1, 0, 0);
  const bonePos = new THREE.Vector3();
  bone.getWorldPosition(bonePos);
  const childPos = new THREE.Vector3();
  childBones[0].getWorldPosition(childPos);
  const dir = childPos.sub(bonePos);
  if (dir.lengthSq() < 1e-6) return new THREE.Vector3(1, 0, 0);
  dir.normalize();
  const worldUp = new THREE.Vector3(0, 1, 0);
  const axis = new THREE.Vector3().crossVectors(dir, worldUp);
  if (axis.lengthSq() < 1e-4) {
    // Bone points straight up/down — swing around world X as fallback.
    return new THREE.Vector3(1, 0, 0);
  }
  return axis.normalize();
}

// Axis perpendicular to the bone direction AND its swing axis.
// For an arm pointing down, this is the "raise arm sideways" axis.
function boneLiftAxis(bone) {
  const childBones = bone.children.filter(c => c.isBone);
  if (childBones.length === 0) return new THREE.Vector3(0, 0, 1);
  const bonePos = new THREE.Vector3();
  bone.getWorldPosition(bonePos);
  const childPos = new THREE.Vector3();
  childBones[0].getWorldPosition(childPos);
  const dir = childPos.sub(bonePos).normalize();
  const swing = boneSwingAxis(bone);
  const axis = new THREE.Vector3().crossVectors(swing, dir);
  if (axis.lengthSq() < 1e-4) return new THREE.Vector3(0, 0, 1);
  return axis.normalize();
}

function worldRotationTrack(bone, worldAxis, anglesFn, duration, steps) {
  const times = makeTimes(duration, steps);
  const values = new Float32Array((steps + 1) * 4);
  const qRest = bone.quaternion.clone();
  const localAxis = localAxisFromWorld(bone, worldAxis);
  for (let i = 0; i <= steps; i++) {
    const angle = anglesFn(i / steps);
    const delta = new THREE.Quaternion().setFromAxisAngle(localAxis, angle);
    const q = delta.multiply(qRest);
    values[i * 4 + 0] = q.x;
    values[i * 4 + 1] = q.y;
    values[i * 4 + 2] = q.z;
    values[i * 4 + 3] = q.w;
  }
  return new THREE.QuaternionKeyframeTrack(`${bone.name}.quaternion`, times, values);
}

function positionTrack(bone, offsetFn, duration, steps) {
  const times = makeTimes(duration, steps);
  const values = new Float32Array((steps + 1) * 3);
  const p0 = bone.position.clone();
  for (let i = 0; i <= steps; i++) {
    const o = offsetFn(i / steps);
    values[i * 3 + 0] = p0.x + o.x;
    values[i * 3 + 1] = p0.y + o.y;
    values[i * 3 + 2] = p0.z + o.z;
  }
  return new THREE.VectorKeyframeTrack(`${bone.name}.position`, times, values);
}

function scaleTrack(bone, factorFn, duration, steps) {
  const times = makeTimes(duration, steps);
  const values = new Float32Array((steps + 1) * 3);
  const s0 = bone.scale.clone();
  for (let i = 0; i <= steps; i++) {
    const f = factorFn(i / steps);
    values[i * 3 + 0] = s0.x * f;
    values[i * 3 + 1] = s0.y * f;
    values[i * 3 + 2] = s0.z * f;
  }
  return new THREE.VectorKeyframeTrack(`${bone.name}.scale`, times, values);
}

const AX_X = new THREE.Vector3(1, 0, 0);
const AX_Y = new THREE.Vector3(0, 1, 0);
const AX_Z = new THREE.Vector3(0, 0, 1);

// === Clip builders ===========================================================

function buildIdleSwayClip(skel) {
  const duration = 3.2, steps = 48;
  const tracks = [];
  // Subtle spine sway (Y rotation) — if we have a spine, sway it; else sway root.
  const target = skel.spine && skel.spine[0] || skel.rootBone;
  tracks.push(worldRotationTrack(target, AX_Y, t => Math.sin(t * Math.PI * 2) * 0.08, duration, steps));
  // Gentle root bob
  tracks.push(positionTrack(skel.rootBone, t => ({ x: 0, y: Math.sin(t * Math.PI * 2) * 0.01, z: 0 }), duration, steps));
  return new THREE.AnimationClip('Idle', duration, tracks);
}

function buildBreatheClip(skel) {
  const duration = 3.0, steps = 40;
  const tracks = [];
  // Scale the lower spine bones outward slightly — avoid scaling root (whole body) or head.
  const chest = skel.spine ? skel.spine.slice(0, Math.min(3, skel.spine.length)) : [skel.rootBone];
  chest.forEach((bone, i) => {
    const amp = 0.03 * (1.0 - i * 0.2);
    tracks.push(scaleTrack(bone, t => 1 + Math.sin(t * Math.PI * 2) * amp, duration, steps));
  });
  return new THREE.AnimationClip('Breathe', duration, tracks);
}

function buildSpinClip(skel) {
  const duration = 4.0, steps = 64;
  const tracks = [worldRotationTrack(skel.rootBone, AX_Y, t => t * Math.PI * 2, duration, steps)];
  return new THREE.AnimationClip('Spin', duration, tracks);
}

function buildWalkClip(skel) {
  if (skel.legs.length < 2) return null;
  const duration = 1.0, steps = 32;
  const tracks = [];

  skel.legs.forEach((leg, legIdx) => {
    const isRightLeg = legIdx >= skel.legs.length / 2;
    const phaseOffset = isRightLeg ? Math.PI : 0;
    leg.forEach((bone, depth) => {
      const swingAxis = boneSwingAxis(bone);  // bone-relative fwd/back axis
      if (depth === 0) {
        tracks.push(worldRotationTrack(bone, swingAxis,
          t => Math.sin(t * Math.PI * 2 + phaseOffset) * 0.6, duration, steps));
      } else if (depth === 1) {
        // Knee bends: peak flexion during swing-through (leg forward).
        tracks.push(worldRotationTrack(bone, swingAxis,
          t => Math.max(0, -Math.sin(t * Math.PI * 2 + phaseOffset)) * 0.9,
          duration, steps));
      } else if (depth === 2) {
        // Ankle follow-through.
        tracks.push(worldRotationTrack(bone, swingAxis,
          t => Math.sin(t * Math.PI * 2 + phaseOffset) * 0.2, duration, steps));
      }
    });
  });

  skel.arms.forEach((arm, armIdx) => {
    const isRightArm = armIdx >= skel.arms.length / 2;
    const phaseOffset = isRightArm ? 0 : Math.PI;
    arm.slice(0, 2).forEach((bone, depth) => {
      const swingAxis = boneSwingAxis(bone);
      const amp = depth === 0 ? 0.45 : 0.25;
      tracks.push(worldRotationTrack(bone, swingAxis,
        t => Math.sin(t * Math.PI * 2 + phaseOffset) * amp, duration, steps));
    });
  });

  // Root bob per footfall (twice per cycle).
  tracks.push(positionTrack(skel.rootBone,
    t => ({ x: 0, y: Math.abs(Math.sin(t * Math.PI * 2)) * 0.04, z: 0 }),
    duration, steps));

  return new THREE.AnimationClip('Walk', duration, tracks);
}

function buildSitClip(skel) {
  if (skel.legs.length < 2) return null;
  const duration = 2.5, steps = 50;
  const tracks = [];
  const easeDown = t => {
    if (t < 0.25) return Math.sin((t / 0.25) * Math.PI * 0.5);
    if (t > 0.75) return Math.sin(((1 - t) / 0.25) * Math.PI * 0.5);
    return 1;
  };
  tracks.push(positionTrack(skel.rootBone,
    t => ({ x: 0, y: -0.2 * easeDown(t), z: 0 }), duration, steps));
  if (skel.spine && skel.spine.length > 0) {
    const spineAxis = boneSwingAxis(skel.spine[0]);
    tracks.push(worldRotationTrack(skel.spine[0], spineAxis,
      t => 0.2 * easeDown(t), duration, steps));
  }
  skel.legs.forEach(leg => {
    if (leg[0]) {
      const axis = boneSwingAxis(leg[0]);
      tracks.push(worldRotationTrack(leg[0], axis,
        t => 1.1 * easeDown(t), duration, steps));
    }
    if (leg[1]) {
      const axis = boneSwingAxis(leg[1]);
      tracks.push(worldRotationTrack(leg[1], axis,
        t => -1.5 * easeDown(t), duration, steps));
    }
  });
  return new THREE.AnimationClip('Sit', duration, tracks);
}

function buildJumpClip(skel) {
  if (skel.legs.length < 2) return null;
  const duration = 1.4, steps = 42;
  const tracks = [];
  const crouch = t => {
    if (t < 0.2) return t / 0.2;
    if (t < 0.3) return 1 - (t - 0.2) / 0.1;
    if (t > 0.8 && t < 0.9) return (t - 0.8) / 0.1;
    if (t >= 0.9) return 1 - (t - 0.9) / 0.1;
    return 0;
  };
  const airY = t => (t > 0.3 && t < 0.8) ? Math.sin(((t - 0.3) / 0.5) * Math.PI) * 0.5 : 0;
  tracks.push(positionTrack(skel.rootBone,
    t => ({ x: 0, y: airY(t) - 0.2 * crouch(t), z: 0 }), duration, steps));
  skel.legs.forEach(leg => {
    if (leg[0]) {
      const a = boneSwingAxis(leg[0]);
      tracks.push(worldRotationTrack(leg[0], a,
        t => 0.7 * crouch(t), duration, steps));
    }
    if (leg[1]) {
      const a = boneSwingAxis(leg[1]);
      tracks.push(worldRotationTrack(leg[1], a,
        t => -1.2 * crouch(t), duration, steps));
    }
  });
  return new THREE.AnimationClip('Jump', duration, tracks);
}

function buildWaveClip(skel) {
  if (skel.arms.length < 1) return null;
  const arm = skel.arms[skel.arms.length - 1];
  if (arm.length < 1) return null;
  const duration = 2.4, steps = 48;
  const tracks = [];
  const raise = t => {
    if (t < 0.2) return t / 0.2;
    if (t > 0.8) return 1 - (t - 0.8) / 0.2;
    return 1;
  };
  // Shoulder: lift arm outward (around shoulder's lift axis).
  const shoulder = arm[0];
  const liftAxis = boneLiftAxis(shoulder);
  tracks.push(worldRotationTrack(shoulder, liftAxis,
    t => -1.6 * raise(t), duration, steps));
  // Elbow: flex inward (around elbow's swing axis).
  if (arm[1]) {
    const elbowAxis = boneSwingAxis(arm[1]);
    tracks.push(worldRotationTrack(arm[1], elbowAxis,
      t => -1.0 * raise(t), duration, steps));
  }
  // Hand: waggle side-to-side.
  const handBone = arm[arm.length - 1];
  const handAxis = boneLiftAxis(handBone);
  tracks.push(worldRotationTrack(handBone, handAxis,
    t => Math.sin((t - 0.2) / 0.6 * Math.PI * 5) * 0.5 * raise(t), duration, steps));
  return new THREE.AnimationClip('Wave', duration, tracks);
}

function buildWiggleClip(skel) {
  const duration = 1.6, steps = 48, amp = 0.08;
  const tracks = [];
  skel.bones.forEach((bone, i) => {
    const phase = (i * 0.37) % (Math.PI * 2);
    const axis = new THREE.Vector3(
      Math.sin(i * 1.3), Math.cos(i * 0.7), Math.sin(i * 2.1)
    ).normalize();
    tracks.push(worldRotationTrack(bone, axis,
      t => Math.sin(t * Math.PI * 2 + phase) * amp,
      duration, steps));
  });
  return new THREE.AnimationClip('Wiggle', duration, tracks);
}

function buildDanceClip(skel) {
  const duration = 1.6, steps = 48, amp = 0.12;
  const tracks = [];
  // Root: syncopated hip bob + sway
  tracks.push(positionTrack(skel.rootBone,
    t => ({ x: 0, y: Math.abs(Math.sin(t * Math.PI * 4)) * amp * 1.5, z: 0 }),
    duration, steps));
  tracks.push(worldRotationTrack(skel.rootBone, AX_Y,
    t => Math.sin(t * Math.PI * 2) * amp * 2.0,
    duration, steps));
  // Arms: wave up and down, alternating
  skel.arms.forEach((arm, ai) => {
    const phase = ai * Math.PI;
    const shoulder = arm[0];
    if (shoulder) tracks.push(worldRotationTrack(shoulder, AX_Z,
      t => -0.4 + Math.sin(t * Math.PI * 2 + phase) * 0.35,
      duration, steps));
  });
  // Other bones: phase-offset shimmy
  skel.bones.forEach((bone, i) => {
    if (bone === skel.rootBone) return;
    if (skel.arms.some(a => a[0] === bone)) return;
    const phase = (i * 0.5) % (Math.PI * 2);
    const axis = new THREE.Vector3(
      0.3 * Math.sin(i * 0.9), 0.7, 0.3 * Math.cos(i * 1.1)
    ).normalize();
    const localAmp = amp * (0.3 + 0.5 * ((i % 5) / 5));
    tracks.push(worldRotationTrack(bone, axis,
      t => Math.sin(t * Math.PI * 2 + phase) * localAmp,
      duration, steps));
  });
  return new THREE.AnimationClip('Dance', duration, tracks);
}

function buildRunClip(skel) {
  // Faster walk with bigger amplitude and stronger arm swing.
  if (skel.legs.length < 2) return null;
  const duration = 0.6, steps = 30;
  const tracks = [];
  skel.legs.forEach((leg, legIdx) => {
    const isRight = legIdx >= skel.legs.length / 2;
    const phase = isRight ? Math.PI : 0;
    leg.forEach((bone, depth) => {
      const axis = boneSwingAxis(bone);
      if (depth === 0) tracks.push(worldRotationTrack(bone, axis,
        t => Math.sin(t * Math.PI * 2 + phase) * 0.95, duration, steps));
      else if (depth === 1) tracks.push(worldRotationTrack(bone, axis,
        t => Math.max(0, -Math.sin(t * Math.PI * 2 + phase)) * 1.6, duration, steps));
      else if (depth === 2) tracks.push(worldRotationTrack(bone, axis,
        t => Math.sin(t * Math.PI * 2 + phase) * 0.3, duration, steps));
    });
  });
  skel.arms.forEach((arm, ai) => {
    const isRight = ai >= skel.arms.length / 2;
    const phase = isRight ? 0 : Math.PI;
    arm.slice(0, 2).forEach((bone, depth) => {
      const axis = boneSwingAxis(bone);
      const amp = depth === 0 ? 0.9 : 0.6;
      tracks.push(worldRotationTrack(bone, axis,
        t => Math.sin(t * Math.PI * 2 + phase) * amp, duration, steps));
    });
  });
  tracks.push(positionTrack(skel.rootBone,
    t => ({ x: 0, y: Math.abs(Math.sin(t * Math.PI * 2)) * 0.08, z: 0 }),
    duration, steps));
  return new THREE.AnimationClip('Run', duration, tracks);
}

function buildCrouchClip(skel) {
  if (skel.legs.length < 2) return null;
  const duration = 2.4, steps = 40;
  const tracks = [];
  const ease = t => {
    if (t < 0.2) return t / 0.2;
    if (t > 0.8) return 1 - (t - 0.8) / 0.2;
    return 1;
  };
  tracks.push(positionTrack(skel.rootBone,
    t => ({ x: 0, y: -0.12 * ease(t), z: 0 }), duration, steps));
  skel.legs.forEach(leg => {
    if (leg[0]) {
      const a = boneSwingAxis(leg[0]);
      tracks.push(worldRotationTrack(leg[0], a,
        t => 0.4 * ease(t), duration, steps));
    }
    if (leg[1]) {
      const a = boneSwingAxis(leg[1]);
      tracks.push(worldRotationTrack(leg[1], a,
        t => -0.7 * ease(t), duration, steps));
    }
  });
  return new THREE.AnimationClip('Crouch', duration, tracks);
}

function buildProceduralClips(object) {
  const skel = analyzeSkeleton(object);
  if (!skel) return [];
  const clips = [];
  clips.push(buildIdleSwayClip(skel));
  clips.push(buildBreatheClip(skel));
  clips.push(buildSpinClip(skel));
  const walk = buildWalkClip(skel);     if (walk) clips.push(walk);
  const run = buildRunClip(skel);       if (run) clips.push(run);
  const wave = buildWaveClip(skel);     if (wave) clips.push(wave);
  const sit = buildSitClip(skel);       if (sit) clips.push(sit);
  const crouch = buildCrouchClip(skel); if (crouch) clips.push(crouch);
  const jump = buildJumpClip(skel);     if (jump) clips.push(jump);
  clips.push(buildWiggleClip(skel));
  clips.push(buildDanceClip(skel));
  return clips;
}

function applyWireframe() {
  if (!currentModel) return;
  currentModel.traverse(o => {
    if (o.isMesh && o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach(m => { m.wireframe = tgWire.checked; });
    }
  });
}

tgSkel.addEventListener('change', () => { if (skeletonHelper) skeletonHelper.visible = tgSkel.checked; });
tgWire.addEventListener('change', applyWireframe);
tgGrid.addEventListener('change', () => { grid.visible = tgGrid.checked; });
const tgClass = document.getElementById('tg-class');
if (tgClass) {
  tgClass.addEventListener('change', () => {
    if (classificationHelpers) classificationHelpers.visible = tgClass.checked;
  });
}

const lgAmbient = document.getElementById('lg-ambient');
const lgKey = document.getElementById('lg-key');
const lgFill = document.getElementById('lg-fill');
const lgRim = document.getElementById('lg-rim');
const lgBg = document.getElementById('lg-bg');
const envPreset = document.getElementById('env-preset');

const lgExposure = document.getElementById('lg-exposure');

function applyLighting() {
  ambientLight.intensity = parseFloat(lgAmbient.value);
  keyLight.intensity = parseFloat(lgKey.value);
  fillLight.intensity = parseFloat(lgFill.value);
  rimLight.intensity = parseFloat(lgRim.value);
  if (lgExposure) renderer.toneMappingExposure = parseFloat(lgExposure.value);
  scene.background = new THREE.Color(lgBg.value);
}

function applyPreset(name) {
  const p = ENV_PRESETS[name];
  if (!p) return;
  lgAmbient.value = p.ambient;
  lgKey.value = p.key;
  lgFill.value = p.fill;
  lgRim.value = p.rim;
  if (lgExposure && p.exposure != null) lgExposure.value = p.exposure;
  keyLight.color.setHex(p.keyColor);
  fillLight.color.setHex(p.fillColor);
  rimLight.color.setHex(p.rimColor);
  lgBg.value = '#' + p.bg.toString(16).padStart(6, '0');
  applyLighting();
}

[lgAmbient, lgKey, lgFill, lgRim, lgBg, lgExposure].filter(Boolean).forEach(el =>
  el.addEventListener('input', applyLighting)
);
envPreset.addEventListener('change', () => applyPreset(envPreset.value));
applyPreset('studio');

function setStatus(msg, busy=false) {
  if (busy) {
    statusText.innerHTML = `<span class="spinner"></span> ${msg}`;
  } else {
    statusText.textContent = msg;
  }
}

async function cancelActive() {
  if (!activeJobId) return;
  cancelBtn.disabled = true;
  try {
    const r = await fetch('/api/cancel/' + activeJobId, { method: 'POST' });
    const d = await r.json();
    setStatus(`Cancel requested: ${d.note || ''}`, true);
  } catch (e) {
    setStatus('Cancel failed: ' + e.message);
  }
}
cancelBtn.addEventListener('click', cancelActive);

let uploadedImageFile = null;

const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const uploadFilename = document.getElementById('upload-filename');
const clearUploadBtn = document.getElementById('clear-upload');

uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  setUploadedImage(f);
});
clearUploadBtn.addEventListener('click', () => setUploadedImage(null));

function setUploadedImage(file) {
  uploadedImageFile = file;
  if (file) {
    uploadFilename.textContent = file.name;
    uploadStatus.style.display = 'flex';
    uploadBtn.style.background = '#2e5090';
    uploadBtn.style.borderColor = '#4a90e2';
    uploadBtn.textContent = '🖼 ✓';
    // Show preview immediately
    const reader = new FileReader();
    reader.onload = (ev) => {
      previewImg.src = ev.target.result;
      previewImg.style.display = 'block';
      previewWrap.classList.remove('empty');
    };
    reader.readAsDataURL(file);
    promptEl.placeholder = 'uploaded image will be used (prompt ignored)';
    setStatus(`Image loaded: ${file.name}. Pick a model and click Generate.`);
  } else {
    uploadedImageFile = null;
    uploadStatus.style.display = 'none';
    uploadBtn.style.background = '';
    uploadBtn.style.borderColor = '';
    uploadBtn.textContent = '🖼';
    fileInput.value = '';
    previewImg.style.display = 'none';
    previewWrap.classList.add('empty');
    promptEl.placeholder = 'e.g. a robot dog with antennae, cartoon style (or upload an image →)';
    setStatus('Ready.');
  }
  mirrorUploadToMvFront();
  if (typeof updateGenViewsBtn === 'function') updateGenViewsBtn();
}

// When the conventional upload is used AND the multi-view model is selected,
// visually place the uploaded image in the "front" slot so it's clear that
// Zero123++ will fill the other 3 from it.
function mirrorUploadToMvFront() {
  const drop = document.querySelector('.mv-drop[data-view="front"]');
  if (!drop) return;
  const isMv = modelSelect.value === 'hunyuan3dmv';
  // If user manually filled the front slot, leave it alone.
  if (mvFiles.front) return;
  if (isMv && uploadedImageFile) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      drop.style.backgroundImage = `url("${ev.target.result}")`;
      drop.classList.add('filled');
    };
    reader.readAsDataURL(uploadedImageFile);
  } else {
    drop.style.backgroundImage = '';
    drop.classList.remove('filled');
  }
}

// Multi-view slots (only used when model = hunyuan3dmv)
const mvFiles = { front: null, back: null, left: null, right: null };

function bindMvSlot(view) {
  const drop = document.querySelector(`.mv-drop[data-view="${view}"]`);
  const input = document.querySelector(`input[type="file"][data-view="${view}"]`);
  if (!drop || !input) return;
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) setMvFile(view, f);
  });
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove('drag');
  }));
  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f && f.type.startsWith('image/')) setMvFile(view, f);
  });
}
function setMvFile(view, file) {
  mvFiles[view] = file;
  const drop = document.querySelector(`.mv-drop[data-view="${view}"]`);
  if (!drop) return;
  if (file) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      drop.style.backgroundImage = `url("${ev.target.result}")`;
      drop.classList.add('filled');
    };
    reader.readAsDataURL(file);
  } else {
    drop.style.backgroundImage = '';
    drop.classList.remove('filled');
  }
}
['front', 'back', 'left', 'right'].forEach(bindMvSlot);

// Pose gallery picker
let selectedPose = '';

async function loadPoseGallery() {
  try {
    const r = await fetch('/api/poses');
    const d = await r.json();
    const grid = document.getElementById('pose-grid');
    if (!grid) return;
    grid.innerHTML = '';
    if (!d.ready) {
      grid.innerHTML = '<div style="grid-column: 1 / -1; font-size: 11px; color: #666;">ControlNet unavailable — ' + (d.error || '') + '</div>';
      return;
    }
    d.poses.forEach(p => {
      const tile = document.createElement('div');
      tile.className = 'pose-tile';
      tile.dataset.pose = p.key;
      tile.style.backgroundImage = `url("${p.url}")`;
      tile.title = p.label;
      tile.innerHTML = `<div class="label">${p.label}</div>`;
      tile.addEventListener('click', () => {
        if (selectedPose === p.key) {
          selectedPose = '';
        } else {
          selectedPose = p.key;
        }
        document.querySelectorAll('.pose-tile').forEach(t => {
          t.classList.toggle('selected', t.dataset.pose === selectedPose);
        });
      });
      grid.appendChild(tile);
    });
  } catch (e) { console.warn('pose gallery load failed', e); }
}
loadPoseGallery();

const genViewsBtn = document.getElementById('gen-views-btn');

function updateGenViewsBtn() {
  // Button is usable only when mv model is selected AND we have a front
  // image (either in the mv slot or via the conventional upload).
  const isMv = modelSelect.value === 'hunyuan3dmv';
  const hasFront = !!mvFiles.front || !!uploadedImageFile;
  const needsOthers = !(mvFiles.back && mvFiles.left && mvFiles.right);
  if (!isMv) {
    genViewsBtn.style.display = 'none';
    return;
  }
  genViewsBtn.style.display = 'block';
  genViewsBtn.disabled = !(hasFront && needsOthers);
  if (!hasFront) {
    genViewsBtn.style.color = '#555';
    genViewsBtn.title = 'Upload a front view first (via the 🖼 button or front slot).';
  } else if (!needsOthers) {
    genViewsBtn.style.color = '#888';
    genViewsBtn.textContent = '🎲 Views already filled (clear a slot to regenerate)';
  } else {
    genViewsBtn.style.color = '#ddd';
    genViewsBtn.textContent = '🎲 Generate other views (Zero123++)';
  }
}

async function generateViews() {
  const sourceFile = mvFiles.front || uploadedImageFile;
  if (!sourceFile) return;
  genViewsBtn.disabled = true;
  genViewsBtn.textContent = '🎲 Synthesizing views (Zero123++)...';
  setStatus('Zero123++ generating back/left/right views...', true);
  try {
    const form = new FormData();
    form.append('image', sourceFile);
    form.append('seed', Math.floor(Math.random() * 1e9).toString());
    const resp = await fetch('/api/generate_views', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'view generation failed');

    // Fetch each generated view back as a Blob so FormData submissions work
    for (const view of ['back', 'left', 'right']) {
      if (!data.urls[view] || mvFiles[view]) continue;
      const blobResp = await fetch(data.urls[view]);
      const blob = await blobResp.blob();
      const file = new File([blob], `${view}.png`, { type: 'image/png' });
      setMvFile(view, file);
    }
    // Also mirror the front if empty
    if (!mvFiles.front && data.urls.front) {
      const blobResp = await fetch(data.urls.front);
      const blob = await blobResp.blob();
      const file = new File([blob], 'front.png', { type: 'image/png' });
      setMvFile('front', file);
    }
    setStatus('Views ready. Click Generate to build the 3D mesh.');
  } catch (e) {
    setStatus('View gen error: ' + e.message);
  } finally {
    updateGenViewsBtn();
  }
}

genViewsBtn.addEventListener('click', generateViews);

// Re-check the button state whenever slot state could have changed
const _origSetMv = setMvFile;
setMvFile = function(view, file) {
  _origSetMv(view, file);
  updateGenViewsBtn();
};

// Rig-editor click handler: raycast against classification spheres.
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (!classificationHelpers || !classificationHelpers.visible) return;
  if (e.button !== 0) return;
  // Skip if the user clicked on the gizmo itself (let TransformControls handle).
  if (transformControls.axis) return;
  const rect = renderer.domElement.getBoundingClientRect();
  mouseNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouseNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouseNDC, camera);
  const hits = raycaster.intersectObjects(classificationHelpers.children, true);
  console.log('[rig-edit] click hits=', hits.length);
  if (hits.length === 0) return;
  const bone = hits[0].object.userData.bone;
  if (bone) {
    console.log('[rig-edit] attaching to', bone.name);
    attachEditorToBone(bone);
  }
});

function attachEditorToBone(bone) {
  editingBone = bone;
  transformControls.attach(bone);
  const info = document.getElementById('edit-info');
  if (info) info.textContent = `Editing: ${bone.name}`;
}

function detachEditor() {
  editingBone = null;
  transformControls.detach();
  const info = document.getElementById('edit-info');
  if (info) info.textContent = 'Click a classification sphere to edit a bone.';
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') detachEditor();
  if (e.key === 'r' || e.key === 'R') transformControls.setMode('rotate');
  if (e.key === 't' || e.key === 'T') transformControls.setMode('translate');
});

async function exportEditedGLB() {
  if (!currentModel) return;
  const exporter = new GLTFExporter();
  const arr = await new Promise((resolve, reject) => {
    exporter.parse(currentModel, resolve, reject, { binary: true, onlyVisible: false });
  });
  const blob = new Blob([arr], { type: 'model/gltf-binary' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'edited_rig.glb';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
window.exportEditedGLB = exportEditedGLB;

// Drag-and-drop onto viewer area
const viewerDropEl = viewerEl;
['dragenter', 'dragover'].forEach(ev => viewerDropEl.addEventListener(ev, (e) => {
  e.preventDefault(); viewerDropEl.style.outline = '2px dashed #4a90e2';
}));
['dragleave', 'drop'].forEach(ev => viewerDropEl.addEventListener(ev, (e) => {
  e.preventDefault(); viewerDropEl.style.outline = '';
}));
viewerDropEl.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith('image/')) setUploadedImage(f);
});

async function generate() {
  const prompt = promptEl.value.trim();
  const isMvModel = modelSelect.value === 'hunyuan3dmv';
  const hasMvSlots = !!mvFiles.front;
  // Use the multi-view slots endpoint if mv slots are filled.
  // Use the single-image endpoint if the conventional upload button was used
  // (server auto-runs Zero123++ if model is hunyuan3dmv).
  const usingMv = isMvModel && hasMvSlots;
  const usingUpload = !usingMv && !!uploadedImageFile;

  if (isMvModel && !hasMvSlots && !uploadedImageFile && !prompt) {
    setStatus('For multi-view: upload a front image (either via the 🖼 button or into a slot), or enter a text prompt.');
    return;
  }
  if (!isMvModel && !usingUpload && !prompt) return;

  goBtn.disabled = true;
  cancelBtn.style.display = 'inline-block';
  cancelBtn.disabled = false;
  if (!usingUpload && !usingMv) {
    previewImg.style.display = 'none';
    previewWrap.classList.add('empty');
  }
  downloadBtn.classList.add('disabled');
  jobInfo.textContent = '';
  setStatus('Submitting...', true);

  let resp, data;
  try {
    if (usingMv) {
      const form = new FormData();
      form.append('image_front', mvFiles.front);
      if (mvFiles.back) form.append('image_back', mvFiles.back);
      if (mvFiles.left) form.append('image_left', mvFiles.left);
      if (mvFiles.right) form.append('image_right', mvFiles.right);
      form.append('settings', JSON.stringify(userSettings));
      form.append('note', prompt);
      resp = await fetch('/api/generate_multiview', { method: 'POST', body: form });
    } else if (usingUpload) {
      const form = new FormData();
      form.append('image', uploadedImageFile);
      form.append('model', modelSelect.value);
      form.append('settings', JSON.stringify(userSettings));
      form.append('note', prompt);
      resp = await fetch('/api/generate_image', { method: 'POST', body: form });
    } else {
      resp = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, model: modelSelect.value, style: styleSelect.value, pose: selectedPose, settings: userSettings })
      });
    }
    data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'submit failed');
  } catch (e) {
    setStatus('Error: ' + e.message);
    goBtn.disabled = false;
    return;
  }

  const jobId = data.job_id;
  activeJobId = jobId;
  jobInfo.innerHTML = `<span class="k">job</span> ${jobId}<br><span class="k">model</span> ${modelSelect.value}<br><span class="k">style</span> ${styleSelect.value}`;

  const start = Date.now();
  let previewShown = false;
  while (true) {
    await new Promise(r => setTimeout(r, 1200));
    let s;
    try {
      s = await fetch('/api/status/' + jobId).then(r => r.json());
    } catch (e) {
      setStatus('Poll error: ' + e.message);
      continue;
    }
    const elapsed = ((Date.now() - start) / 1000).toFixed(0);
    if (s.status === 'queued') {
      setStatus(`Queued (${elapsed}s)...`, true);
    } else if (s.status === 'image') {
      setStatus(usingUpload ? `Processing image... (${elapsed}s)` : `Generating 2D image... (${elapsed}s)`, true);
    } else if (s.status === 'views') {
      setStatus(`Synthesizing novel views with Zero123++... (${elapsed}s)`, true);
      if (s.views_urls && !previewShown) {
        // show front view in the main preview box
        if (s.views_urls.front) {
          previewImg.src = s.views_urls.front + '?t=' + Date.now();
          previewImg.style.display = 'block';
          previewWrap.classList.remove('empty');
          previewShown = true;
        }
      }
      // Fill the mv-drop slots with generated views for user feedback
      if (s.views_urls) {
        ['front','back','left','right'].forEach(v => {
          const drop = document.querySelector(`.mv-drop[data-view="${v}"]`);
          if (drop && s.views_urls[v] && !drop.classList.contains('filled')) {
            drop.style.backgroundImage = `url("${s.views_urls[v]}")`;
            drop.classList.add('filled');
          }
        });
      }
    } else if (s.status === 'mesh') {
      if (s.preview_url && !previewShown) {
        previewImg.src = s.preview_url + '?t=' + Date.now();
        previewImg.style.display = 'block';
        previewWrap.classList.remove('empty');
        previewShown = true;
      }
      setStatus(`Generating 3D mesh... (${elapsed}s, ~30s typical)`, true);
    } else if (s.status === 'done') {
      setStatus(`Done in ${elapsed}s.`);
      cancelBtn.style.display = 'none';
      activeJobId = null;
      if (s.preview_url && !previewShown) {
        previewImg.src = s.preview_url;
        previewImg.style.display = 'block';
        previewWrap.classList.remove('empty');
      }
      downloadBtn.href = s.glb_url;
      downloadBtn.classList.remove('disabled');
      jobInfo.innerHTML = `<span class="k">job</span> ${jobId}<br><span class="k">model</span> ${s.model || '?'}<br><span class="k">t2i</span> ${s.t2i || '?'}<br><span class="k">style</span> ${s.style || '?'}<br><span class="k">seed</span> ${s.seed ?? '?'}<br><span class="k">time</span> ${elapsed}s`;
      loadGLB(s.glb_url);
      goBtn.disabled = false;
      return;
    } else if (s.status === 'cancelled') {
      setStatus(`Cancelled after ${elapsed}s.`);
      cancelBtn.style.display = 'none';
      activeJobId = null;
      jobInfo.innerHTML = `<span class="k">job</span> ${jobId} (cancelled)`;
      goBtn.disabled = false;
      return;
    } else if (s.status === 'error') {
      setStatus('Error: ' + s.error);
      cancelBtn.style.display = 'none';
      activeJobId = null;
      jobInfo.innerHTML = `<span class="k">job</span> ${jobId} (failed)`;
      goBtn.disabled = false;
      return;
    }
  }
}

goBtn.addEventListener('click', generate);
promptEl.addEventListener('keydown', e => { if (e.key === 'Enter') generate(); });
