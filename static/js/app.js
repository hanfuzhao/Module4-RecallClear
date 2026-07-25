/* RecallClear front-end.
   Plain ES modules-free JavaScript so the page works without a build step and
   without any third-party script. */

(function () {
  "use strict";

  var CARD_SECTIONS = [
    "WHAT'S WRONG",
    "WHAT COULD HAPPEN",
    "HOW URGENT",
    "WHAT TO DO",
    "WHAT IT COSTS"
  ];

  var URGENCY_CLASSES = {
    "STOP DRIVING": "level-stop",
    "PARK OUTSIDE": "level-park",
    "GET IT FIXED SOON": "level-soon",
    "SCHEDULE IT": "level-schedule"
  };

  var elements = {};
  var examples = [];
  var currentRecord = null;
  var isBusy = false;

  /* -- helpers ------------------------------------------------------------ */

  function byId(id) {
    return document.getElementById(id);
  }

  function show(element) {
    if (element) { element.classList.remove("is-hidden"); }
  }

  function hide(element) {
    if (element) { element.classList.add("is-hidden"); }
  }

  function setStatus(message, isError, isBusyState) {
    var line = elements.status;
    if (!line) { return; }
    line.className = "help" + (isError ? " status-error" : "");
    line.innerHTML = "";
    if (isBusyState) {
      var spinner = document.createElement("span");
      spinner.className = "spinner";
      line.appendChild(spinner);
    }
    line.appendChild(document.createTextNode(message || ""));
  }

  function setBusy(busy, message) {
    isBusy = busy;
    elements.explainBtn.disabled = busy;
    if (elements.lookupBtn) { elements.lookupBtn.disabled = busy; }
    setStatus(message || "", false, busy);
  }

  function postJson(url, payload) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json().then(function (data) {
        if (!response.ok) { throw new Error(data.error || "Request failed."); }
        return data;
      });
    });
  }

  /* -- tabs --------------------------------------------------------------- */

  function activateTab(tabId, panelId) {
    var tabs = document.querySelectorAll(".tab");
    for (var i = 0; i < tabs.length; i += 1) {
      var isTarget = tabs[i].id === tabId;
      tabs[i].classList.toggle("is-active", isTarget);
      tabs[i].setAttribute("aria-selected", isTarget ? "true" : "false");
    }
    var panels = document.querySelectorAll(".tab-panel");
    for (var j = 0; j < panels.length; j += 1) {
      panels[j].classList.toggle("is-hidden", panels[j].id !== panelId);
    }
  }

  /* -- rendering ---------------------------------------------------------- */

  function renderOfficialWarnings(warnings) {
    var container = elements.officialBanners;
    container.innerHTML = "";
    if (!warnings || (!warnings.do_not_drive && !warnings.fire_risk_when_parked)) {
      hide(elements.officialPanel);
      return;
    }
    if (warnings.do_not_drive) {
      container.appendChild(buildBanner(
        "banner-danger", "⛔", "NHTSA says: do not drive this vehicle",
        "This is the agency's own do-not-drive designation for this campaign, not the model's opinion. Contact your dealer before driving."
      ));
    }
    if (warnings.fire_risk_when_parked) {
      container.appendChild(buildBanner(
        "banner-warn", "🔥", "NHTSA says: park outside, away from buildings",
        "This campaign carries a fire risk even when the vehicle is parked and switched off."
      ));
    }
    show(elements.officialPanel);
  }

  function buildBanner(cssClass, icon, title, body) {
    var banner = document.createElement("div");
    banner.className = "banner " + cssClass;

    var iconSpan = document.createElement("span");
    iconSpan.className = "banner-icon";
    iconSpan.setAttribute("aria-hidden", "true");
    iconSpan.textContent = icon;

    var textWrap = document.createElement("div");
    var strong = document.createElement("strong");
    strong.textContent = title;
    var paragraph = document.createElement("p");
    paragraph.textContent = body;
    textWrap.appendChild(strong);
    textWrap.appendChild(paragraph);

    banner.appendChild(iconSpan);
    banner.appendChild(textWrap);
    return banner;
  }

  function renderCard(result, notice, noticeMetrics) {
    var sections = result.sections || {};

    var urgencyText = sections["HOW URGENT"] || "";
    var level = "";
    for (var key in URGENCY_CLASSES) {
      if (urgencyText.toUpperCase().indexOf(key) === 0) { level = key; break; }
    }
    if (!level) {
      for (var fallback in URGENCY_CLASSES) {
        if (urgencyText.toUpperCase().indexOf(fallback) >= 0) { level = fallback; break; }
      }
    }

    elements.cardUrgency.className = "card-urgency " + (URGENCY_CLASSES[level] || "");
    elements.cardUrgency.innerHTML = "";
    if (level) {
      var label = document.createElement("span");
      label.textContent = level;
      elements.cardUrgency.appendChild(label);
      var reason = document.createElement("span");
      reason.className = "reason";
      reason.textContent = urgencyText.slice(level.length).replace(/^[\s–—-]+/, "");
      elements.cardUrgency.appendChild(reason);
    } else {
      elements.cardUrgency.textContent = urgencyText || "Urgency not stated";
    }

    elements.cardBody.innerHTML = "";
    CARD_SECTIONS.forEach(function (section) {
      if (section === "HOW URGENT") { return; }
      var term = document.createElement("dt");
      term.textContent = section;
      var definition = document.createElement("dd");
      if (sections[section]) {
        definition.textContent = sections[section];
      } else {
        definition.textContent = "The model did not produce this section.";
        definition.className = "is-missing";
      }
      elements.cardBody.appendChild(term);
      elements.cardBody.appendChild(definition);
    });

    elements.readability.innerHTML = "";
    appendMetric(elements.readability, "Original notice", noticeMetrics.grade_level + " grade");
    appendMetric(elements.readability, "Rewrite", result.grade_level + " grade");
    appendMetric(elements.readability, "Generated in", result.seconds + "s");

    elements.rawNotice.textContent = notice;
    elements.rawOutput.textContent = result.raw;
    show(elements.resultPanel);
  }

  function appendMetric(container, label, value) {
    var span = document.createElement("span");
    span.appendChild(document.createTextNode(label + " "));
    var bold = document.createElement("b");
    bold.textContent = value;
    span.appendChild(bold);
    container.appendChild(span);
  }

  function renderComparison(baseResult, tunedResult) {
    if (!baseResult) {
      hide(elements.baselinePanel);
      return;
    }
    elements.baselineOutput.textContent = baseResult.raw || "(empty output)";
    elements.tunedOutput.textContent = tunedResult.raw || "(empty output)";
    fillMiniMetrics(elements.baselineMetrics, baseResult);
    fillMiniMetrics(elements.tunedMetrics, tunedResult);
    show(elements.baselinePanel);
  }

  function fillMiniMetrics(list, result) {
    list.innerHTML = "";
    addMiniMetric(list, "All five sections", result.well_formed ? "yes" : "no");
    addMiniMetric(list, "Reading grade", String(result.grade_level));
    addMiniMetric(list, "Jargon per 100 words", String(result.jargon_rate));
    addMiniMetric(list, "Words", String((result.raw || "").split(/\s+/).filter(Boolean).length));
    addMiniMetric(list, "Seconds", String(result.seconds));
  }

  function addMiniMetric(list, label, value) {
    var item = document.createElement("li");
    var name = document.createElement("span");
    name.textContent = label;
    var bold = document.createElement("b");
    bold.textContent = value;
    item.appendChild(name);
    item.appendChild(bold);
    list.appendChild(item);
  }

  /* -- actions ------------------------------------------------------------ */

  function currentNotice() {
    return (elements.noticeInput.value || "").trim();
  }

  function handleLookup() {
    var campaign = (elements.campaignInput.value || "").trim();
    if (!campaign) {
      setStatus("Enter a campaign number such as 23V123000.", true, false);
      return;
    }
    setBusy(true, "Fetching " + campaign.toUpperCase() + " from NHTSA ...");
    postJson("/api/lookup", { campaign_number: campaign })
      .then(function (data) {
        currentRecord = data.record;
        elements.noticeInput.value = data.notice;
        renderOfficialWarnings(data.official_warnings);
        activateTab("tab-paste", "panel-paste");
        setBusy(false, "Loaded " + data.record.campaign_number + " — " + data.record.manufacturer + ".");
      })
      .catch(function (error) {
        setBusy(false, "");
        setStatus(error.message, true, false);
      });
  }

  function handleExplain() {
    var notice = currentNotice();
    if (!notice) {
      setStatus("Paste a notice or look one up first.", true, false);
      return;
    }
    var includeBaseline = elements.compareToggle.checked;
    setBusy(true, includeBaseline
      ? "Running the base model and the fine-tuned model ..."
      : "Rewriting the notice ...");
    hide(elements.baselinePanel);

    postJson("/api/explain", { notice: notice, include_baseline: includeBaseline })
      .then(function (data) {
        renderCard(data.tuned, notice, data.notice_metrics);
        renderComparison(data.base, data.tuned);
        setBusy(false, "Done. The original notice is always shown below the summary.");
        elements.resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      })
      .catch(function (error) {
        setBusy(false, "");
        setStatus(error.message, true, false);
      });
  }

  function handleExampleClick(event) {
    var button = event.target.closest("[data-example-index]");
    if (!button || isBusy) { return; }
    var example = examples[Number(button.getAttribute("data-example-index"))];
    if (!example) { return; }
    elements.noticeInput.value = example.notice;
    currentRecord = null;
    renderOfficialWarnings(null);
    activateTab("tab-paste", "panel-paste");
    setStatus(example.manufacturer + " — " + example.campaign_number + " loaded. Press “Explain this recall”.", false, false);
  }

  function checkHealth() {
    fetch("/api/health")
      .then(function (response) { return response.json(); })
      .then(function (data) {
        if (data.status === "ok") { elements.healthDot.classList.add("is-ready"); }
      })
      .catch(function () { /* health is cosmetic; ignore failures */ });
  }

  /* -- wiring ------------------------------------------------------------- */

  function init() {
    elements = {
      status: byId("status-line"),
      explainBtn: byId("explain-btn"),
      lookupBtn: byId("lookup-btn"),
      campaignInput: byId("campaign-input"),
      noticeInput: byId("notice-input"),
      compareToggle: byId("compare-toggle"),
      officialPanel: byId("official-panel"),
      officialBanners: byId("official-banners"),
      resultPanel: byId("result-panel"),
      cardUrgency: byId("card-urgency"),
      cardBody: byId("card-body"),
      readability: byId("readability-strip"),
      rawNotice: byId("raw-notice"),
      rawOutput: byId("raw-output"),
      baselinePanel: byId("baseline-panel"),
      baselineOutput: byId("baseline-output"),
      tunedOutput: byId("tuned-output"),
      baselineMetrics: byId("baseline-metrics"),
      tunedMetrics: byId("tuned-metrics"),
      healthDot: byId("health-dot")
    };

    var payload = byId("examples-data");
    if (payload) {
      try { examples = JSON.parse(payload.textContent) || []; } catch (error) { examples = []; }
    }

    elements.explainBtn.addEventListener("click", handleExplain);
    if (elements.lookupBtn) {
      elements.lookupBtn.addEventListener("click", handleLookup);
      elements.campaignInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter") { handleLookup(); }
      });
    }
    document.addEventListener("click", handleExampleClick);

    var tabs = document.querySelectorAll(".tab");
    for (var i = 0; i < tabs.length; i += 1) {
      tabs[i].addEventListener("click", function (event) {
        activateTab(event.currentTarget.id, event.currentTarget.getAttribute("aria-controls"));
      });
    }

    checkHealth();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
