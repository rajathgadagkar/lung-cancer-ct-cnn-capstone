(() => {
  "use strict";

  // ---------------------------------------------------------------------------
  // 1. Theme Toggle
  // ---------------------------------------------------------------------------
  const root = document.documentElement;
  const themeBtn = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");

  const applyTheme = (theme) => {
    root.setAttribute("data-theme", theme);
    themeIcon.textContent = theme === "dark" ? "☀️" : "🌙";
    localStorage.setItem("lungcnn-theme", theme);
  };

  applyTheme(localStorage.getItem("lungcnn-theme") || "dark");
  themeBtn.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
  });

  // ---------------------------------------------------------------------------
  // 2. Interactive CT Preprocessing & Watershed Pipeline Explorer
  // ---------------------------------------------------------------------------
  const PIPELINE_STAGES = [
    {
      short: "Block Diagram",
      badge: "Architecture Overview",
      title: "End-to-End System Block Diagram",
      image: "assets/figures/block_diagram.png",
      description:
        "Raw chest CT DICOM slices pass through three sequential noise-reduction & contrast stages, followed by Otsu thresholding + marker-controlled Watershed segmentation, and finally binary classification (Benign vs. Malignant) via our custom 12-layer CNN.",
      code: "DICOM CT -> Median Filter -> Gaussian Filter -> CLAHE\n         -> Otsu Threshold -> Watershed Segmentation -> 12-Layer CNN",
      rationale:
        "Combining classical medical image processing with deep convolutional feature extraction eliminates scanner artifacts and highlights nodule boundaries before classification."
    },
    {
      short: "Denoise + AHE",
      badge: "Stage 1–3 · Filtering",
      title: "Original -> Median Filter -> Gaussian Filter -> Adaptive Histogram Equalization",
      image: "assets/figures/preprocessing_pipeline.png",
      description:
        "Median filtering removes impulsive salt-and-pepper noise without blurring edges; Gaussian filtering (sigma = 2) suppresses high-frequency acquisition noise; Adaptive Histogram Equalization (CLAHE) restores local tile contrast reduced by Gaussian smoothing.",
      code: "MATLAB: Q = medfilt2(I); Z = imgaussfilt(Q, 2); E = adapthisteq(Z);\nPython: cv2.medianBlur -> ndimage.gaussian_filter(sigma=2) -> cv2.createCLAHE()",
      rationale:
        "Low-dose CT scans frequently contain speckle and impulse noise that trigger false positives. Restoring contrast via tile-level AHE makes soft-tissue nodules stand out clearly against dark lung parenchyma."
    },
    {
      short: "Otsu Binary",
      badge: "Stage 4 · Thresholding",
      title: "Global Otsu Thresholding (Binarized CT Slice)",
      image: "assets/figures/seg_01_binarized.png",
      description:
        "Global Otsu thresholding computes an optimal intensity threshold T separating foreground anatomical structures and candidate nodules from the dark lung cavity.",
      code: "MATLAB: level = graythresh(Z); BW = imbinarize(Z, level);\nPython: thresh = skimage.filters.threshold_otsu(clahe_img); bw = clahe_img > thresh",
      rationale:
        "Fast, parameter-free foreground/background separation that prepares a clean binary mask for Euclidean distance transformation."
    },
    {
      short: "Dist Transform",
      badge: "Stage 5 · Distance Map",
      title: "Complement, Inverse Distance Transform & Negative Infinity Mask",
      image: "assets/figures/seg_02_complement_distance.png",
      description:
        "The binary mask is complemented so candidate nodules form basins, transformed via negative Euclidean distance (-bwdist), and foreground pixels are forced to -Inf so only true object interiors act as local minima.",
      code: "MATLAB: D = -bwdist(~BW); D(~BW) = -Inf;\nPython: D = -ndimage.distance_transform_edt(~bw); D[~bw] = -np.inf",
      rationale:
        "Translates binary shapes into a smooth topographic elevation surface suitable for watershed flooding."
    },
    {
      short: "Extended Minima",
      badge: "Stage 6 · Over-Segmentation Control",
      title: "Extended-Minima Grayscale Mask (imextendedmin)",
      image: "assets/figures/seg_03_grayscale_mask.png",
      description:
        "Filters out shallow, spurious local minima (h = 2) and imposes only significant regional minima via morphological reconstruction (imimposemin).",
      code: "MATLAB: mask = imextendedmin(D, 2); D2 = imimposemin(D, mask);\nPython: local_min = skimage.morphology.h_minima(finite_dist, h=2.0)",
      rationale:
        "Prevents classic watershed over-segmentation so nodules are segmented as coherent structures instead of dozens of noisy fragments."
    },
    {
      short: "Watershed Lines",
      badge: "Stage 7 · Boundaries",
      title: "Watershed Ridge Lines Superimposed on Binary Mask",
      image: "assets/figures/seg_04_watershed_lines.png",
      description:
        "Watershed flooding draws 1-pixel-thick ridge dams (Ld == 0) wherever adjacent catchment basins meet, cleanly separating touching or overlapping pulmonary structures.",
      code: "MATLAB: Ld = watershed(D2); bw2 = BW; bw2(Ld == 0) = 0;\nPython: labels = skimage.segmentation.watershed(finite_dist, markers, watershed_line=True)",
      rationale:
        "Produces continuous, closed object contours even when nodules lie adjacent to the chest wall or pulmonary vessels."
    },
    {
      short: "RGB Labels",
      badge: "Stage 8 · Region Map",
      title: "Watershed Region Labels (label2rgb Colormap)",
      image: "assets/figures/seg_05_watershed_labels.png",
      description:
        "Each distinct catchment basin returned by the watershed label matrix is rendered in a unique color to verify region separation.",
      code: "MATLAB: rgbLabels = label2rgb(Ld, 'hot', 'w');",
      rationale:
        "Visually confirms that distinct anatomical regions and candidate nodules receive independent region IDs."
    },
    {
      short: "Final Composite",
      badge: "Stage 9 · CNN Input",
      title: "Final Watershed Segmented Composite Output",
      image: "assets/figures/seg_06_watershed_output.png",
      description:
        "Fuses the watershed-delimited binary mask with the contrast-enhanced CT slice via difference compositing (imfuse) and resizes to 512×512 (256×256 for CNN input).",
      code: "MATLAB: temp = imfuse(bw2, Z, 'diff'); out = imresize(im2double(temp), [512 512]);\nPython: fused = np.abs(watershed_binary.astype(np.float32) - clahe_img / 255.0)",
      rationale:
        "Preserves internal grayscale nodule texture while embedding crisp watershed boundary contours for the CNN."
    },
    {
      short: "vs K-Means (K=2)",
      badge: "Ablation Comparison",
      title: "Comparison: K-Means Clustering (K = 2)",
      image: "assets/figures/seg_07_kmeans_k2.png",
      description:
        "Evaluated K-Means clustering with K=2 clusters as an alternative segmentation baseline.",
      code: "Baseline comparison: K-Means clustering (K = 2 intensity clusters)",
      rationale:
        "K-Means groups pixels solely by intensity without topological boundary separation, failing to isolate touching nodules."
    },
    {
      short: "vs K-Means (K=3)",
      badge: "Ablation Comparison",
      title: "Comparison: K-Means Clustering (K = 3)",
      image: "assets/figures/seg_08_kmeans_k3.png",
      description:
        "K-Means with K=3 separates bone, soft tissue, and air, but still cannot draw distinct contours around individual overlapping nodules.",
      code: "Baseline comparison: K-Means clustering (K = 3 intensity clusters)",
      rationale:
        "Confirms our capstone conclusion: marker-controlled Watershed outperforms K-Means for lung nodule boundary delineation."
    }
  ];

  let currentStage = 0;
  const stageTabs = document.getElementById("stageTabs");
  const stageImage = document.getElementById("stageImage");
  const stageBadge = document.getElementById("stageBadge");
  const stageTitle = document.getElementById("stageTitle");
  const stageDescription = document.getElementById("stageDescription");
  const stageCode = document.getElementById("stageCode");
  const stageRationale = document.getElementById("stageRationale");
  const stageCounter = document.getElementById("stageCounter");

  const renderStage = (idx) => {
    currentStage = (idx + PIPELINE_STAGES.length) % PIPELINE_STAGES.length;
    const s = PIPELINE_STAGES[currentStage];
    stageImage.src = s.image;
    stageImage.alt = s.title;
    stageBadge.textContent = s.badge;
    stageTitle.textContent = s.title;
    stageDescription.textContent = s.description;
    stageCode.textContent = s.code;
    stageRationale.textContent = s.rationale;
    stageCounter.textContent = `${currentStage + 1} / ${PIPELINE_STAGES.length}`;
    stageTabs.querySelectorAll("button").forEach((btn, i) => {
      btn.classList.toggle("active", i === currentStage);
    });
  };

  PIPELINE_STAGES.forEach((s, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tab-btn";
    btn.textContent = `${i + 1}. ${s.short}`;
    btn.addEventListener("click", () => renderStage(i));
    stageTabs.appendChild(btn);
  });

  document.getElementById("prevStageBtn").addEventListener("click", () => renderStage(currentStage - 1));
  document.getElementById("nextStageBtn").addEventListener("click", () => renderStage(currentStage + 1));
  renderStage(0);

  // ---------------------------------------------------------------------------
  // 3. Interactive 14x Data Augmentation Lab (HTML5 Canvas)
  // ---------------------------------------------------------------------------
  const AUG_PRESETS = [
    { name: "#1 Original", scale: 1.0, shear: 0, hFlip: false, vFlip: false },
    { name: "#2 Sheared", scale: 1.0, shear: 14, hFlip: false, vFlip: false },
    { name: "#3 Scaled", scale: 0.92, shear: 0, hFlip: false, vFlip: false },
    { name: "#4 H-Flip", scale: 1.0, shear: 0, hFlip: true, vFlip: false },
    { name: "#5 V-Flip", scale: 1.0, shear: 0, hFlip: false, vFlip: true },
    { name: "#6 Sheared + H-Flip", scale: 1.0, shear: 14, hFlip: true, vFlip: false },
    { name: "#7 Scaled + H-Flip", scale: 0.92, shear: 0, hFlip: true, vFlip: false },
    { name: "#8 V-Flip + H-Flip", scale: 1.0, shear: 0, hFlip: true, vFlip: true },
    { name: "#9 Sheared + V-Flip", scale: 1.0, shear: 14, hFlip: false, vFlip: true },
    { name: "#10 Scaled + V-Flip", scale: 0.92, shear: 0, hFlip: false, vFlip: true },
    { name: "#11 H-Flip + Scaled + Sheared", scale: 0.93, shear: 12, hFlip: true, vFlip: false },
    { name: "#12 All 4 Combined", scale: 0.92, shear: 15, hFlip: true, vFlip: true },
    { name: "#13 V-Flip + Scaled + Sheared", scale: 0.93, shear: 12, hFlip: false, vFlip: true },
    { name: "#14 Sheared + Scaled", scale: 0.92, shear: 14, hFlip: false, vFlip: false }
  ];

  const augCanvas = document.getElementById("augCanvas");
  const ctx = augCanvas.getContext("2d");
  const scaleSlider = document.getElementById("scaleSlider");
  const shearSlider = document.getElementById("shearSlider");
  const hFlipToggle = document.getElementById("hFlipToggle");
  const vFlipToggle = document.getElementById("vFlipToggle");
  const scaleVal = document.getElementById("scaleVal");
  const shearVal = document.getElementById("shearVal");
  const augActivePreset = document.getElementById("augActivePreset");
  const augMatrixText = document.getElementById("augMatrixText");
  const presetGrid = document.getElementById("presetGrid");

  const ctSample = new Image();
  ctSample.src = "assets/figures/seg_06_watershed_output.png";

  const drawAugmented = (presetLabel = "Custom Transform") => {
    const scale = parseFloat(scaleSlider.value);
    const shearDeg = parseFloat(shearSlider.value);
    const hFlip = hFlipToggle.checked;
    const vFlip = vFlipToggle.checked;

    scaleVal.textContent = `${scale.toFixed(2)}×`;
    shearVal.textContent = `${shearDeg}°`;
    augActivePreset.textContent = presetLabel;
    augMatrixText.textContent =
      `Scale: ${scale.toFixed(2)}× · Shear: ${shearDeg}° · H-Flip: ${hFlip ? "On" : "Off"} · V-Flip: ${vFlip ? "On" : "Off"}`;

    const w = augCanvas.width;
    const h = augCanvas.height;
    ctx.save();
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, w, h);

    if (ctSample.complete && ctSample.naturalWidth > 0) {
      ctx.translate(w / 2, h / 2);
      const sx = (hFlip ? -1 : 1) * scale;
      const sy = (vFlip ? -1 : 1) * scale;
      const shx = Math.tan((shearDeg * Math.PI) / 180);
      ctx.transform(sx, 0, shx * sy, sy, 0, 0);
      ctx.drawImage(ctSample, -w / 2, -h / 2, w, h);
    }
    ctx.restore();
  };

  ctSample.addEventListener("load", () => drawAugmented("#1 Original"));

  AUG_PRESETS.forEach((p, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `preset-btn${idx === 0 ? " active" : ""}`;
    btn.textContent = p.name;
    btn.addEventListener("click", () => {
      scaleSlider.value = String(p.scale);
      shearSlider.value = String(p.shear);
      hFlipToggle.checked = p.hFlip;
      vFlipToggle.checked = p.vFlip;
      presetGrid.querySelectorAll("button").forEach((b, i) => b.classList.toggle("active", i === idx));
      drawAugmented(p.name);
    });
    presetGrid.appendChild(btn);
  });

  [scaleSlider, shearSlider, hFlipToggle, vFlipToggle].forEach((el) => {
    el.addEventListener("input", () => {
      presetGrid.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      drawAugmented("Custom Transform");
    });
  });

  document.getElementById("resetAugBtn").addEventListener("click", () => {
    presetGrid.querySelector("button").click();
  });

  // ---------------------------------------------------------------------------
  // 4. Interactive 12-Layer Proposed CNN Inspector
  // ---------------------------------------------------------------------------
  const LAYERS = [
    ["conv2d", "Conv2D", "(None, 62, 62, 96)", "11,712", "96 filters · 11×11 kernel · stride 4 · valid · ReLU", "Large 11×11 receptive field with stride 4 rapidly captures broad lung anatomy from 256×256×1 CT slices while cutting spatial cost."],
    ["batch_normalization", "BatchNormalization", "(None, 62, 62, 96)", "384", "96 channels (gamma, beta, moving mean/var)", "Reduces internal covariate shift (ICS) and accelerates convergence after Conv1."],
    ["max_pooling2d", "MaxPool2D", "(None, 30, 30, 96)", "0", "3×3 pool · stride 2", "Downsamples feature maps to 30×30 while retaining peak nodule activations."],
    ["conv2d_1", "Conv2D", "(None, 30, 30, 256)", "614,656", "256 filters · 5×5 kernel · stride 1 · same · ReLU", "Extracts mid-level nodule texture and contour features with same padding."],
    ["batch_normalization_1", "BatchNormalization", "(None, 30, 30, 256)", "1,024", "256 channels", "Stabilizes activations across mini-batches of 20 CT slices."],
    ["max_pooling2d_1", "MaxPool2D", "(None, 14, 14, 256)", "0", "3×3 pool · stride 2", "Reduces spatial resolution to 14×14."],
    ["dropout", "Dropout", "(None, 14, 14, 256)", "0", "Rate = 0.7", "Aggressive 0.7 dropout between convolutional blocks prevents memorization of augmented scans."],
    ["conv2d_2", "Conv2D", "(None, 14, 14, 384)", "885,120", "384 filters · 3×3 kernel · stride 1 · same · ReLU", "First layer of the 3-conv series block extracting fine-grained nodule morphology."],
    ["batch_normalization_2", "BatchNormalization", "(None, 14, 14, 384)", "1,536", "384 channels", "Normalizes 384 feature maps."],
    ["conv2d_3", "Conv2D", "(None, 14, 14, 384)", "1,327,488", "384 filters · 3×3 kernel · stride 1 · same · ReLU", "Deepens spatial representation at 14×14 resolution."],
    ["batch_normalization_3", "BatchNormalization", "(None, 14, 14, 384)", "1,536", "384 channels", "Maintains stable gradient flow in the mid-network block."],
    ["conv2d_4", "Conv2D", "(None, 14, 14, 256)", "98,560", "256 filters · 1×1 kernel · stride 1 · same · ReLU", "1×1 bottleneck projection preserves spatial map size while compressing channels from 384 to 256."],
    ["batch_normalization_4", "BatchNormalization", "(None, 14, 14, 256)", "1,024", "256 channels", "Normalizes projected 256-channel feature tensor."],
    ["max_pooling2d_2", "MaxPool2D", "(None, 6, 6, 256)", "0", "3×3 pool · stride 2", "Condenses features to 6×6 spatial grid."],
    ["dropout_1", "Dropout", "(None, 6, 6, 256)", "0", "Rate = 0.7", "Regularizes deep convolutional features before the 512-filter block."],
    ["conv2d_5", "Conv2D", "(None, 6, 6, 512)", "3,277,312", "512 filters · 5×5 kernel · stride 1 · same · ReLU", "High-capacity 512-filter layer capturing complex semantic malignancy patterns."],
    ["batch_normalization_5", "BatchNormalization", "(None, 6, 6, 512)", "2,048", "512 channels", "Accelerates convergence on high-dimensional 512-channel maps."],
    ["conv2d_6", "Conv2D", "(None, 6, 6, 512)", "6,554,112", "512 filters · 5×5 kernel · stride 1 · same · ReLU", "Largest convolutional block (6.55M params) refining high-level lesion cues."],
    ["batch_normalization_6", "BatchNormalization", "(None, 6, 6, 512)", "2,048", "512 channels", "Normalizes 512-channel activations."],
    ["conv2d_7", "Conv2D", "(None, 6, 6, 384)", "196,992", "384 filters · 1×1 kernel · stride 1 · same · ReLU", "1×1 convolution compacts channels from 512 to 384 before final pooling, keeping flattened vector concise (1,536 vs 2,048)."],
    ["batch_normalization_7", "BatchNormalization", "(None, 6, 6, 384)", "1,536", "384 channels", "Normalizes final convolutional output."],
    ["max_pooling2d_3", "MaxPool2D", "(None, 2, 2, 384)", "0", "3×3 pool · stride 2", "Final spatial pooling down to 2×2×384."],
    ["dropout_2", "Dropout", "(None, 2, 2, 384)", "0", "Rate = 0.7", "Regularizes spatial embeddings prior to flattening."],
    ["flatten", "Flatten", "(None, 1536)", "0", "2 × 2 × 384 = 1,536 features", "Converts 3D spatial feature map into a 1D feature vector for discriminative dense layers."],
    ["dense", "Dense", "(None, 4096)", "6,295,552", "4,096 units · ReLU", "First fully connected classifier layer: (1536 + 1) × 4096 = 6,295,552 params."],
    ["dropout_3", "Dropout", "(None, 4096)", "0", "Rate = 0.7", "Prevents co-adaptation across 4,096 hidden neurons."],
    ["dense_1", "Dense", "(None, 4096)", "16,781,312", "4,096 units · ReLU", "Second fully connected discriminative layer: (4096 + 1) × 4096 = 16,781,312 params."],
    ["dropout_4", "Dropout", "(None, 4096)", "0", "Rate = 0.7", "Strong regularization between high-capacity dense layers."],
    ["dense_2", "Dense", "(None, 4096)", "16,781,312", "4,096 units · ReLU", "Third fully connected layer refining final class-separable representation."],
    ["dropout_5", "Dropout", "(None, 4096)", "0", "Rate = 0.7", "Final dropout before the 2-class softmax output."],
    ["dense_3", "Dense", "(None, 2)", "8,194", "2 units · Softmax", "Outputs calibrated probabilities for Benign (0) vs. Malignant (1)."]
  ];

  const dotClass = (type) => {
    if (type === "Conv2D") return "dot-conv";
    if (type === "BatchNormalization") return "dot-bn";
    if (type === "MaxPool2D") return "dot-pool";
    if (type === "Dropout") return "dot-drop";
    return "dot-dense";
  };

  const layerStack = document.getElementById("layerStack");
  const selectLayer = (idx) => {
    const [name, type, shape, params, cfg, why] = LAYERS[idx];
    document.getElementById("inspType").textContent = type;
    document.getElementById("inspIndex").textContent = `Layer ${idx + 1} / ${LAYERS.length}`;
    document.getElementById("inspName").textContent = name;
    document.getElementById("inspShape").textContent = shape;
    document.getElementById("inspParams").textContent = params;
    document.getElementById("inspConfig").textContent = cfg;
    document.getElementById("inspWhy").textContent = why;
    layerStack.querySelectorAll("button").forEach((b, i) => b.classList.toggle("active", i === idx));
  };

  LAYERS.forEach(([name, type, shape, params], idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "layer-row";
    btn.innerHTML = `<i class="dot ${dotClass(type)}"></i><strong>${name}</strong><span class="mono-muted">${shape}</span><span class="mono-muted" style="text-align:right">${params}</span>`;
    btn.addEventListener("click", () => selectLayer(idx));
    layerStack.appendChild(btn);
  });
  selectLayer(0);

  // ---------------------------------------------------------------------------
  // 5. Interactive Benchmark Dashboard & Curves
  // ---------------------------------------------------------------------------
  const BENCH_DATA = [
    { name: "Proposed 12-Layer CNN", accuracy: 99.28, valLoss: 0.0235, f1: 0.99, errors: 1, winner: true },
    { name: "AlexNet (8 Layers)", accuracy: 89.99, valLoss: 0.9657, f1: 0.91, errors: 14, winner: false },
    { name: "LeNet (5 Layers)", accuracy: 88.57, valLoss: 0.9127, f1: 0.88, errors: 17, winner: false },
    { name: "VGG16 (16 Layers)", accuracy: 51.43, valLoss: 0.6930, f1: 0.00, errors: 75, winner: false }
  ];

  const METRIC_META = {
    accuracy: {
      heading: "Test Accuracy Comparison (%)",
      max: 100,
      fmt: (v) => `${v.toFixed(2)}%`,
      insight: "Proposed 12-Layer CNN outperforms AlexNet by +9.29% and LeNet by +10.71%, while VGG16 collapsed on the small medical corpus."
    },
    valLoss: {
      heading: "Validation Loss (Lower is Better)",
      max: 1.0,
      fmt: (v) => v.toFixed(4),
      insight: "BatchNormalization after every Conv2D + 0.7 Dropout reduced validation loss to 0.0235 (vs. 0.91–0.97 overfitting loss on LeNet/AlexNet)."
    },
    f1: {
      heading: "Malignant Class (1) F1-Score",
      max: 1.0,
      fmt: (v) => v.toFixed(2),
      insight: "Proposed CNN achieved 1.00 precision and 0.99 recall on the malignant class (F1 = 0.99), compared to 0.91 for AlexNet and 0.88 for LeNet."
    },
    errors: {
      heading: "Misclassifications out of 140 Test Scans (Lower is Better)",
      max: 80,
      fmt: (v) => `${v} / 140`,
      insight: "Only 1 misclassification on the 140-image evaluation split (and 138/140 on unseen test evaluation), cutting AlexNet's error rate by 14×."
    }
  };

  const benchBars = document.getElementById("benchBars");
  const chartHeading = document.getElementById("chartHeading");
  const chartInsight = document.getElementById("chartInsight");

  const renderBench = (metric) => {
    const meta = METRIC_META[metric];
    chartHeading.textContent = meta.heading;
    chartInsight.textContent = meta.insight;
    benchBars.innerHTML = "";
    BENCH_DATA.forEach((row) => {
      const val = row[metric];
      const pct = Math.max(2, Math.min(100, (val / meta.max) * 100));
      const item = document.createElement("div");
      item.innerHTML = `
        <div class="bar-item-head">
          <span>${row.winner ? "<strong>" + row.name + "</strong>" : row.name}</span>
          <span class="mono-muted">${meta.fmt(val)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill${row.winner ? " winner" : ""}" style="width:${pct}%"></div>
        </div>
      `;
      benchBars.appendChild(item);
    });
  };

  document.getElementById("metricSwitcher").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-metric]");
    if (!btn) return;
    document.querySelectorAll("#metricSwitcher button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    renderBench(btn.dataset.metric);
  });
  renderBench("accuracy");

  const CURVE_MAP = {
    proposed_acc: ["assets/figures/proposed_accuracy_curve.png", "Proposed 12-Layer CNN accuracy curve across 100 epochs"],
    proposed_loss: ["assets/figures/proposed_loss_curve.png", "Proposed 12-Layer CNN loss curve across 100 epochs"],
    proposed_report: ["assets/figures/proposed_classification_report.png", "Proposed 12-Layer CNN classification report"],
    alexnet: ["assets/figures/alexnet_results.png", "AlexNet accuracy, loss, and classification report"],
    lenet: ["assets/figures/lenet_results.png", "LeNet accuracy and loss plots"],
    vgg16: ["assets/figures/vgg16_results.png", "VGG16 accuracy, loss, and classification report"]
  };

  const curveImg = document.getElementById("curveImage");
  document.getElementById("curveTabs").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-curve]");
    if (!btn) return;
    document.querySelectorAll("#curveTabs button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const [src, alt] = CURVE_MAP[btn.dataset.curve];
    curveImg.src = src;
    curveImg.alt = alt;
  });

  // ---------------------------------------------------------------------------
  // 6. Copy BibTeX Button
  // ---------------------------------------------------------------------------
  const copyBtn = document.getElementById("copyBibtexBtn");
  copyBtn.addEventListener("click", async () => {
    const text = document.getElementById("bibtexCode").textContent;
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "Copy BibTeX"), 1800);
    } catch {
      copyBtn.textContent = "Select & Copy";
    }
  });
})();
