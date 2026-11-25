from fastapi.testclient import TestClient

from payments_rag.api.main import app

http = TestClient(app)


def test_page_is_served_with_a_strict_content_security_policy():
    r = http.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp.split("script-src", 1)[1].split(";", 1)[0]
    assert r.headers["x-content-type-options"] == "nosniff"


def test_page_says_questions_are_logged_and_does_not_promise_a_reply():
    html = http.get("/").text.lower()
    assert "logged" in html and "verbatim" in html
    assert "we will reply" not in html and "will contact you" not in html
    assert '<script src="/app.js"' in html and "<script>" not in html  # no inline script


def test_page_script_never_inserts_html():
    r = http.get("/app.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in r.text, sink


def test_page_script_sends_the_body_hash_cloudfront_needs():
    js = http.get("/app.js").text
    assert "x-amz-content-sha256" in js and "crypto.subtle.digest" in js


def test_page_script_shows_flows_escalation_and_sources():
    js = http.get("/app.js").text
    assert "/flows" in js and "escalated" in js and "sources" in js


def test_stylesheet_is_served():
    r = http.get("/app.css")
    assert r.status_code == 200 and "text/css" in r.headers["content-type"]
