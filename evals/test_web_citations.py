"""Browser citation links must open the exact source selected by the API."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


WEB_HTML = Path(__file__).parents[1] / "src" / "nom_rag" / "web" / "index.html"

NODE_CHECK = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');

const html = fs.readFileSync(0, 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, 'inline web script missing');

let onClick;
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    innerHTML: '', textContent: '', style: {},
    classList: { add() {}, toggle() { return false; } },
    addEventListener() {}, setAttribute() {}, focus() {},
  });
  return elements.get(id);
}
const document = {
  getElementById: element,
  addEventListener(type, handler) { if (type === 'click') onClick = handler; },
};
const context = vm.createContext({
  document, fetch: () => new Promise(() => {}), navigator: {}, setTimeout,
});
vm.runInContext(script, context);

const renderAnswer = vm.runInContext('renderAnswer', context);
const rendered = renderAnswer(
  'Primera [NOM-019 §9.9 a)] y segunda [NOM-019 §9.5].', 0
);
assert.match(rendered, /data-ref="NOM-019:9\.9"/);
assert.match(rendered, /data-ref="NOM-019:9\.5"/);
const grouped = renderAnswer('Ambas [NOM-035 §7.9; NOM-019 §9.6].', 0);
const groupedRefs = [...grouped.matchAll(/data-ref="([^"]+)"/g)].map(m => m[1]);
assert.deepEqual(groupedRefs, ['NOM-035:7.9', 'NOM-019:9.6']);
const lowercase = renderAnswer('Ambas [nom-035 §7.9; nom-019 §9.6].', 0);
assert.deepEqual(
  [...lowercase.matchAll(/data-ref="([^"]+)"/g)].map(m => m[1]),
  groupedRefs
);
const inherited = renderAnswer('Dos [NOM-035 §7.9, §7.10].', 0);
assert.deepEqual(
  [...inherited.matchAll(/data-ref="([^"]+)"/g)].map(m => m[1]),
  ['NOM-035:7.9', 'NOM-035:7.10']
);

const sample = {
  citations: [
    { ref: 'NOM-019:9.9', nom: 'NOM-019', text: 'texto 9.9' },
    { ref: 'NOM-019:9.5', nom: 'NOM-019', text: 'texto 9.5' },
    { ref: 'NOM-030:4.8', nom: 'NOM-030', text: 'texto con 4.8' },
    { ref: 'NOM-035:7.9', nom: 'NOM-035', text: 'texto 7.9' },
    { ref: 'NOM-019:9.6', nom: 'NOM-019', text: 'texto 9.6' },
  ],
  retrieved: [
    { chunk_id: 'NOM-019:9.5', text: 'texto 9.5' },
    { chunk_id: 'NOM-019:9.9', text: 'texto 9.9' },
    { chunk_id: 'NOM-030:4.6', text: 'texto con 4.8' },
    { chunk_id: 'NOM-035:7.9', text: 'texto 7.9' },
    { chunk_id: 'NOM-019:9.6', text: 'texto 9.6' },
  ],
};
context.sample = sample;
vm.runInContext('turnData.push(sample); showRetrieval = () => {};', context);

let opened;
const chunks = sample.retrieved.map(c => ({
  dataset: { chunk: c.chunk_id },
  classList: { add() { opened = c.chunk_id; } },
  querySelector() { return { textContent: '' }; },
  scrollIntoView() {},
}));
element('chunks').querySelectorAll = () => chunks;
function clickCitation(ref) {
  opened = undefined;
  const cite = { dataset: { idx: '0', ref } };
  onClick({ target: { closest(selector) { return selector === '.cite' ? cite : null; } } });
  return opened;
}
assert.equal(clickCitation('NOM-019:9.9'), 'NOM-019:9.9');
assert.equal(clickCitation('NOM-019:9.5'), 'NOM-019:9.5');
assert.equal(clickCitation('NOM-030:4.8'), 'NOM-030:4.6');
for (const ref of groupedRefs) assert.equal(clickCitation(ref), ref);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_citation_buttons_open_their_exact_retrieved_chunk():
    completed = subprocess.run(
        ["node", "-e", NODE_CHECK],
        input=WEB_HTML.read_text(encoding="utf-8"),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
