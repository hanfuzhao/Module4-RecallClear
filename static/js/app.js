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
    "STOP DRIVING": "urgency-stop",
    "PARK OUTSIDE": "urgency-park",
    "GET IT FIXED SOON": "urgency-soon",
    "SCHEDULE IT": "urgency-schedule"
  };

  var MODES = ["base", "few_shot", "tuned"];
  var MODE_WAIT_HINT = {
    base: "Stock model thinking — about 20 s on this host ...",
    few_shot: "Feeding it the crib sheet — the big prompt makes this the slow bay, 30–60 s ...",
    tuned: "Fine-tuned model running — about 20 s ..."
  };

  var elements = {};
  var examples = [];
  var currentRecord = null;
  var busyModes = {};
  var warningsShown = false;

  

  function byId(id) { return document.getElementById(id); }
  function show(el) { if (el) { el.classList.remove("is-hidden"); } }
  function hide(el) { if (el) { el.classList.add("is-hidden"); } }

  function setStatus(message, isError) {
    var line = elements.status;
    if (!line) { return; }
    line.className = "help" + (isError ? " status-error" : "");
    line.textContent = message || "";
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

  function currentNotice() {
    return (elements.noticeInput.value || "").trim();
  }

  

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

  

  function renderOfficialWarnings(warnings, source) {
    var container = elements.officialBanners;
    container.innerHTML = "";
    if (!warnings || (!warnings.do_not_drive && !warnings.fire_risk_when_parked)) {
      hide(elements.officialPanel);
      warningsShown = false;
      return;
    }
    var fromText = source === "text";
    if (warnings.do_not_drive) {
      container.appendChild(buildBanner(
        "banner-danger", "⛔",
        fromText ? "This letter says: do not drive this vehicle"
                 : "NHTSA says: do not drive this vehicle",
        fromText
          ? "Detected from the letter's own wording by a rule, not by the model. Call your dealer before driving."
          : "The agency's own do-not-drive designation for this campaign, not any model's opinion. Call your dealer before driving."
      ));
    }
    if (warnings.fire_risk_when_parked) {
      container.appendChild(buildBanner(
        "banner-warn", "🔥",
        fromText ? "This letter says: park outside, away from buildings"
                 : "NHTSA says: park outside, away from buildings",
        fromText
          ? "Detected from the letter's own wording by a rule, not by the model. There is a fire risk even when parked."
          : "This campaign carries a fire risk even while the vehicle is parked and switched off."
      ));
    }
    show(elements.officialPanel);
    warningsShown = true;
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

  

  function bayStatus(mode, message, isError, isBusy) {
    var box = document.querySelector('[data-status="' + mode + '"]');
    if (!box) { return; }
    box.className = "bay-status" + (isError ? " is-error" : "");
    box.innerHTML = "";
    if (isBusy) {
      var spinner = document.createElement("span");
      spinner.className = "spinner";
      box.appendChild(spinner);
    }
    box.appendChild(document.createTextNode(message || ""));
  }

  function setBayBusy(mode, busy) {
    busyModes[mode] = busy;
    var button = document.querySelector('[data-run="' + mode + '"]');
    if (button) { button.disabled = busy; }
    elements.runAllBtn.disabled = MODES.some(function (m) { return busyModes[m]; });
  }

  function urgencySignClass(urgencyText) {
    var upper = (urgencyText || "").toUpperCase();
    for (var key in URGENCY_CLASSES) {
      if (upper.indexOf(key) === 0) { return { level: key, cls: URGENCY_CLASSES[key] }; }
    }
    for (var fallback in URGENCY_CLASSES) {
      if (upper.indexOf(fallback) >= 0) { return { level: fallback, cls: URGENCY_CLASSES[fallback] }; }
    }
    return { level: "", cls: "urgency-none" };
  }

  function renderBayResult(mode, result) {
    var box = document.querySelector('[data-result="' + mode + '"]');
    if (!box) { return; }
    box.innerHTML = "";

    var sections = result.sections || {};
    var urgency = urgencySignClass(sections["HOW URGENT"]);

    var sign = document.createElement("span");
    sign.className = "urgency-sign " + urgency.cls;
    sign.textContent = urgency.level
      ? urgency.level + " — model's call"
      : "No urgency stated";
    box.appendChild(sign);

    var hasAnySection = CARD_SECTIONS.some(function (section) { return sections[section]; });
    if (hasAnySection) {
      var list = document.createElement("dl");
      list.className = "bay-card";
      CARD_SECTIONS.forEach(function (section) {
        if (section === "HOW URGENT") { return; }
        var term = document.createElement("dt");
        term.textContent = section;
        var definition = document.createElement("dd");
        if (sections[section]) {
          definition.textContent = sections[section];
        } else {
          definition.textContent = "— section missing —";
          definition.className = "is-missing";
        }
        list.appendChild(term);
        list.appendChild(definition);
      });
      box.appendChild(list);
    } else {
      var raw = document.createElement("pre");
      raw.className = "bay-raw";
      raw.textContent = result.raw || "(empty output)";
      box.appendChild(raw);
    }

    var verdict = document.createElement("p");
    verdict.className = "bay-verdict";
    var verdictMark = document.createElement("span");
    verdictMark.className = result.well_formed ? "ok" : "bad";
    verdictMark.textContent = result.well_formed ? "✓ all five sections, in order" : "✗ card incomplete";
    verdict.appendChild(verdictMark);
    box.appendChild(verdict);

    var metrics = document.createElement("ul");
    metrics.className = "bay-metrics";
    addMetric(metrics, "grade", String(result.grade_level));
    addMetric(metrics, "jargon/100w", String(result.jargon_rate));
    addMetric(metrics, "prompt", String(result.prompt_tokens) + " tok");
    addMetric(metrics, "time", String(result.seconds) + " s");
    box.appendChild(metrics);

    show(box);
  }

  function addMetric(list, label, value) {
    var item = document.createElement("li");
    item.appendChild(document.createTextNode(label + " "));
    var bold = document.createElement("b");
    bold.textContent = value;
    item.appendChild(bold);
    list.appendChild(item);
  }

  

  function runMode(mode) {
    var notice = currentNotice();
    if (!notice) {
      setStatus("Paste a letter or grab one from the glovebox first.", true);
      return Promise.resolve();
    }
    setBayBusy(mode, true);
    bayStatus(mode, MODE_WAIT_HINT[mode] || "Running ...", false, true);
    hide(document.querySelector('[data-result="' + mode + '"]'));

    return postJson("/api/explain", { notice: notice, mode: mode })
      .then(function (data) {
        renderBayResult(mode, data.result);
        bayStatus(mode, "Done in " + data.result.seconds + " s.", false, false);
        if (!currentRecord && !warningsShown && data.text_warnings) {
          renderOfficialWarnings(data.text_warnings, "text");
        }
        elements.rawNotice.textContent = notice;
        show(elements.sourcePanel);
      })
      .catch(function (error) {
        bayStatus(mode, error.message, true, false);
      })
      .then(function () {
        setBayBusy(mode, false);
      });
  }

  function runAll() {
    setStatus("Racing all three bays — the crib sheet goes last, it is the slow one.", false);
    runMode("tuned")
      .then(function () { return runMode("base"); })
      .then(function () { return runMode("few_shot"); })
      .then(function () { setStatus("Race finished. Compare the bays.", false); });
  }

  function handleLookup() {
    var campaign = (elements.campaignInput.value || "").trim();
    if (!campaign) {
      setStatus("Enter a campaign number such as 23V123000.", true);
      return;
    }
    setStatus("Pulling " + campaign.toUpperCase() + " from NHTSA ...", false);
    postJson("/api/lookup", { campaign_number: campaign })
      .then(function (data) {
        currentRecord = data.record;
        elements.noticeInput.value = data.notice;
        renderOfficialWarnings(data.official_warnings, "official");
        hide(elements.pasteNotice);
        activateTab("tab-paste", "panel-paste");
        setStatus("Loaded " + data.record.campaign_number + " — " + data.record.manufacturer + ". Now pick a bay.", false);
      })
      .catch(function (error) {
        setStatus(error.message, true);
      });
  }

  function handleExampleClick(event) {
    var button = event.target.closest("[data-example-index]");
    if (!button) { return; }
    var example = examples[Number(button.getAttribute("data-example-index"))];
    if (!example) { return; }
    elements.noticeInput.value = example.notice;
    currentRecord = null;
    renderOfficialWarnings(null);
    show(elements.pasteNotice);
    activateTab("tab-paste", "panel-paste");
    setStatus(example.manufacturer + " — " + example.campaign_number + " loaded. Now pick a bay.", false);
  }

  function checkHealth() {
    fetch("/api/health")
      .then(function (response) { return response.json(); })
      .then(function (data) {
        if (data.status === "ok") { elements.healthDot.classList.add("is-ready"); }
      })
      .catch(function () {  });
  }

  

  function init() {
    elements = {
      status: byId("status-line"),
      lookupBtn: byId("lookup-btn"),
      campaignInput: byId("campaign-input"),
      noticeInput: byId("notice-input"),
      officialPanel: byId("official-panel"),
      officialBanners: byId("official-banners"),
      sourcePanel: byId("source-panel"),
      rawNotice: byId("raw-notice"),
      pasteNotice: byId("paste-notice"),
      runAllBtn: byId("run-all-btn"),
      healthDot: byId("health-dot")
    };

    var payload = byId("examples-data");
    if (payload) {
      try { examples = JSON.parse(payload.textContent) || []; } catch (error) { examples = []; }
    }

    MODES.forEach(function (mode) {
      var button = document.querySelector('[data-run="' + mode + '"]');
      if (button) {
        button.addEventListener("click", function () { runMode(mode); });
      }
    });
    elements.runAllBtn.addEventListener("click", runAll);

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
