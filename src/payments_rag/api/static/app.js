"use strict";
// Every value from the server (model output included) is inserted with textContent, never as HTML:
// an injected passage can steer the answer text, but it cannot run script on this page.

const $ = (id) => document.getElementById(id);

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

// CloudFront signs requests to the Lambda Function URL; for POST it needs the body's SHA-256 here.
async function sha256Hex(text) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

const REASONS = {
  low_retrieval_confidence: "no passage matched the question closely enough",
  insufficient_context: "the retrieved passages do not contain the answer",
  all_passages_quarantined: "the retrieved passages failed a safety check",
  no_valid_citation: "the answer did not cite its sources correctly",
  model_declined: "the model declined to answer",
  truncated: "the answer was cut off",
  output_policy_violation: "the answer failed an output check",
};

function showStatus(text, kind) {
  const status = $("status");
  status.textContent = text || "";
  status.className = kind ? `status ${kind}` : "status";
}

function renderAnswer(data) {
  $("result").hidden = false;
  showStatus("");
  $("answer").textContent = data.answer || "";
  $("answer").hidden = !data.answer;
  $("escalation").hidden = !data.escalated;
  $("reason").textContent = REASONS[data.reason] || data.reason || "";

  const list = $("sources");
  list.replaceChildren();
  for (const s of data.sources || []) {
    const item = el("li");
    item.append(el("code", s.id));
    if (s.source && s.source !== "ISO 20022 e-Repository") item.append(el("span", ` (${s.source})`, "muted"));
    if (s.snippet) item.append(el("p", s.snippet, "snippet"));
    list.append(item);
  }
  $("sources-block").hidden = !(data.sources || []).length || data.escalated;
}

async function ask(question) {
  const button = $("ask-button");
  button.disabled = true;
  $("result").hidden = false;
  $("answer").textContent = "";
  $("escalation").hidden = true;
  $("sources-block").hidden = true;
  showStatus("Looking it up… (the first question after a quiet spell can take a few seconds)", "busy");
  try {
    const body = JSON.stringify({ question });
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "content-type": "application/json", "x-amz-content-sha256": await sha256Hex(body) },
      body,
    });
    if (response.status === 429) {
      const wait = response.headers.get("retry-after");
      showStatus(`Too many questions from your connection. Please try again${wait ? ` in ${wait} seconds` : " later"}.`, "error");
      return;
    }
    if (!response.ok) {
      let detail = "";
      try { detail = (await response.json()).detail; } catch (_) { /* not JSON */ }
      showStatus(typeof detail === "string" && detail ? `Sorry: ${detail}.` : `Sorry, something went wrong (${response.status}).`, "error");
      return;
    }
    renderAnswer(await response.json());
  } catch (_) {
    showStatus("Sorry, the service could not be reached. Please try again.", "error");
  } finally {
    button.disabled = false;
  }
}

async function loadFlows() {
  try {
    const response = await fetch("/flows");
    if (!response.ok) return;
    const data = await response.json();
    $("flows-note").textContent = data.source_note || "";
    const container = $("flows");
    for (const flow of data.flows || []) {
      const block = el("details", undefined, "flow");
      block.append(el("summary", flow.title));
      block.append(el("p", flow.summary));
      const steps = el("ol");
      for (const s of flow.steps || []) {
        const li = el("li");
        li.append(el("span", `${s.from} → ${s.to}: `));
        li.append(el("code", s.message));
        if (s.message_name) li.append(el("span", ` (${s.message_name})`, "muted"));
        li.append(el("span", `. ${s.purpose}`));
        steps.append(li);
      }
      block.append(steps);
      for (const note of flow.notes || []) block.append(el("p", note, "note"));
      container.append(block);
    }
  } catch (_) {
    /* the flows list is optional; questions still work */
  }
}

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (question) ask(question);
});
for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    $("question").value = example.textContent;
    ask(example.textContent);
  });
}
loadFlows();
