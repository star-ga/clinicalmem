// Q16.16 BitNet b1.58 forward pass — vanilla JavaScript port.
//
// Bit-identical with engine.bitnet_classifier.classify(). The same pair
// of drug names will produce the same severity_name + repro_hash +
// logits_q16 in this browser as in the server-side Python forward pass.
//
// No external dependencies. No JS frameworks. Loads
// engine/bitnet_weights.json via fetch(); computes everything else
// inline using BigInt for the BLAKE2b 64-bit operations and Number for
// the Q16.16 integer arithmetic (which fits comfortably inside the
// 2^53 safe-integer range).
//
// Surface (window.ClinicalMemBitNet):
//   loadWeights(url='engine/bitnet_weights.json')  -> Promise<weights>
//   classify(drugA, drugB, weights)                -> Promise<{
//     severity, severity_name, logits_q16, feature_hash, repro_hash,
//     weights_id, deterministic_table_match
//   }>
//
// Apache-2.0 — STARGA, Inc.

(function () {
  "use strict";

  // ─── BLAKE2b ────────────────────────────────────────────────────────────
  // Reference: RFC 7693. Compact implementation using BigInt for 64-bit
  // word arithmetic. Output is truncated to digest_size bytes.

  const BLAKE2B_IV = [
    0x6a09e667f3bcc908n, 0xbb67ae8584caa73bn,
    0x3c6ef372fe94f82bn, 0xa54ff53a5f1d36f1n,
    0x510e527fade682d1n, 0x9b05688c2b3e6c1fn,
    0x1f83d9abfb41bd6bn, 0x5be0cd19137e2179n,
  ];

  const BLAKE2B_SIGMA = [
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    [14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3],
    [11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4],
    [7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8],
    [9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13],
    [2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9],
    [12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11],
    [13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10],
    [6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5],
    [10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0],
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    [14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3],
  ];

  const M64 = 0xffffffffffffffffn;

  function rot64(x, n) {
    n = BigInt(n);
    return (((x >> n) | (x << (64n - n))) & M64);
  }

  function G(v, a, b, c, d, x, y) {
    v[a] = (v[a] + v[b] + x) & M64;
    v[d] = rot64(v[d] ^ v[a], 32);
    v[c] = (v[c] + v[d]) & M64;
    v[b] = rot64(v[b] ^ v[c], 24);
    v[a] = (v[a] + v[b] + y) & M64;
    v[d] = rot64(v[d] ^ v[a], 16);
    v[c] = (v[c] + v[d]) & M64;
    v[b] = rot64(v[b] ^ v[c], 63);
  }

  function blake2bCompress(h, m, t, last) {
    const v = new Array(16);
    for (let i = 0; i < 8; i++) v[i] = h[i];
    for (let i = 0; i < 8; i++) v[i + 8] = BLAKE2B_IV[i];
    v[12] = (v[12] ^ (t & M64)) & M64;
    v[13] = (v[13] ^ ((t >> 64n) & M64)) & M64;
    if (last) v[14] = (v[14] ^ M64) & M64;
    for (let i = 0; i < 12; i++) {
      const s = BLAKE2B_SIGMA[i];
      G(v, 0, 4, 8,  12, m[s[0]],  m[s[1]]);
      G(v, 1, 5, 9,  13, m[s[2]],  m[s[3]]);
      G(v, 2, 6, 10, 14, m[s[4]],  m[s[5]]);
      G(v, 3, 7, 11, 15, m[s[6]],  m[s[7]]);
      G(v, 0, 5, 10, 15, m[s[8]],  m[s[9]]);
      G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
      G(v, 2, 7, 8,  13, m[s[12]], m[s[13]]);
      G(v, 3, 4, 9,  14, m[s[14]], m[s[15]]);
    }
    for (let i = 0; i < 8; i++) h[i] = (h[i] ^ v[i] ^ v[i + 8]) & M64;
  }

  function le64(buf, offset) {
    let x = 0n;
    for (let i = 7; i >= 0; i--) {
      x = (x << 8n) | BigInt(buf[offset + i]);
    }
    return x;
  }

  function blake2b(input, digestSize) {
    if (digestSize <= 0 || digestSize > 64) {
      throw new Error("digestSize must be 1..64");
    }
    const h = BLAKE2B_IV.slice();
    h[0] = (h[0] ^ 0x01010000n ^ BigInt(digestSize)) & M64;

    // Pad input to a multiple of 128 bytes
    const inputLen = input.length;
    const padded = new Uint8Array(Math.max(128, Math.ceil(inputLen / 128) * 128));
    padded.set(input);

    const numBlocks = Math.ceil(inputLen / 128) || 1;
    let processed = 0n;

    for (let i = 0; i < numBlocks - 1; i++) {
      processed += 128n;
      const m = new Array(16);
      for (let j = 0; j < 16; j++) m[j] = le64(padded, i * 128 + j * 8);
      blake2bCompress(h, m, processed, false);
    }
    // Final block — process the remainder; t = total input length.
    processed = BigInt(inputLen);
    const m = new Array(16);
    for (let j = 0; j < 16; j++) m[j] = le64(padded, (numBlocks - 1) * 128 + j * 8);
    blake2bCompress(h, m, processed, true);

    // Output the first digestSize bytes (little-endian per word).
    const out = new Uint8Array(digestSize);
    for (let i = 0; i < digestSize; i++) {
      const word = h[Math.floor(i / 8)];
      out[i] = Number((word >> (BigInt(i % 8) * 8n)) & 0xffn);
    }
    return out;
  }

  // ─── Drug-pair feature encoding (must match Python _encode_drug_token) ──

  const TRIT_LOOKUP = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1];

  function encodeDrugToken(rxcuiOrName) {
    // Canonicalize: lowercase, whitespace-collapsed.
    const canonical = rxcuiOrName.trim().toLowerCase().split(/\s+/).join(" ");
    const bytes = new TextEncoder().encode(canonical);
    const digest = blake2b(bytes, 16);
    const out = [];
    for (const byte of digest) {
      out.push(TRIT_LOOKUP[(byte >> 0) & 0xf]);
      out.push(TRIT_LOOKUP[(byte >> 4) & 0xf]);
      out.push(TRIT_LOOKUP[byte & 0xf]);
      out.push(TRIT_LOOKUP[(byte >> 2) & 0xf]);
    }
    return out.slice(0, 64);
  }

  // ─── Q16.16 arithmetic ──────────────────────────────────────────────────

  // Use 2 ** 31 (or explicit literals) — `1 << 31` overflows JS's 32-bit
  // signed bitwise to -2147483648 and breaks both Q16_MIN and Q16_MAX.
  const Q16_ONE = 65536;            // 2 ** 16
  const Q16_MIN = -2147483648;      // -(2 ** 31)
  const Q16_MAX = 2147483647;       // (2 ** 31) - 1

  function q16Clamp(v) {
    if (v > Q16_MAX) return Q16_MAX;
    if (v < Q16_MIN) return Q16_MIN;
    return v;
  }

  function q16DotTernary(activations, ternaryWeights) {
    let acc = 0;
    for (let i = 0; i < activations.length; i++) {
      const w = ternaryWeights[i];
      if (w === 1) acc += activations[i];
      else if (w === -1) acc -= activations[i];
    }
    return q16Clamp(acc);
  }

  function q16Relu(v) {
    return v > 0 ? v : 0;
  }

  // ─── Weights bundle ─────────────────────────────────────────────────────

  async function loadWeights(url) {
    if (!url) url = "engine/bitnet_weights.json";
    const resp = await fetch(url, { cache: "no-cache" });
    if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
    const payload = await resp.json();

    if (payload.hidden_w.length !== 64) throw new Error("hidden_w must be 64 rows");
    if (payload.hidden_b.length !== 64) throw new Error("hidden_b must be 64");
    if (payload.output_w.length !== 5) throw new Error("output_w must be 5 rows");
    if (payload.output_b.length !== 5) throw new Error("output_b must be 5");

    return {
      hidden_w: payload.hidden_w,
      hidden_b: payload.hidden_b,
      output_w: payload.output_w,
      output_b: payload.output_b,
      bundle_id: (payload._meta && payload._meta.bundle_id) || "",
    };
  }

  // ─── SHA-256 helpers (canonical-JSON hash; matches Python json.dumps
  // sort_keys=True, separators=(",", ":")) ────────────────────────────────

  async function sha256Hex(text) {
    const buf = new TextEncoder().encode(text);
    const hash = await crypto.subtle.digest("SHA-256", buf);
    const bytes = new Uint8Array(hash);
    let out = "";
    for (const b of bytes) out += b.toString(16).padStart(2, "0");
    return out;
  }

  // Canonical JSON matches Python's json.dumps(sort_keys=True,
  // separators=(",",":")) for the values we emit (ints, strings, lists
  // of ints, sorted-key dicts).
  function canonicalJson(value) {
    if (value === null) return "null";
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(value);
    if (typeof value === "string") return JSON.stringify(value);
    if (typeof value === "boolean") return value ? "true" : "false";
    if (Array.isArray(value)) return "[" + value.map(canonicalJson).join(",") + "]";
    const keys = Object.keys(value).sort();
    return "{" + keys.map(k => JSON.stringify(k) + ":" + canonicalJson(value[k])).join(",") + "}";
  }

  // ─── Forward pass ───────────────────────────────────────────────────────

  const SEVERITY_NAMES = ["none", "minor", "moderate", "major", "contraindicated"];

  async function classify(drugA, drugB, weights) {
    // Lex-sort the pair.
    const [a, b] = [drugA, drugB].sort();
    const featureA = encodeDrugToken(a);
    const featureB = encodeDrugToken(b);
    const pair = featureA.concat(featureB);
    if (pair.length !== 128) throw new Error(`pair features length ${pair.length} != 128`);

    // feature_hash = SHA-256 over bytes((v + 1) for v in pair)
    // Python: bytes((v + 1) for v in pair_features) → 128-byte buffer
    const featureBytes = new Uint8Array(pair.map(v => v + 1));
    const featureHashBuf = await crypto.subtle.digest("SHA-256", featureBytes);
    const featureHash = Array.from(new Uint8Array(featureHashBuf))
      .map(b => b.toString(16).padStart(2, "0")).join("");

    // Scale ternary → Q16.16
    const activations = pair.map(v => v * Q16_ONE);

    // First linear: 128 → 64, plus bias, then ReLU
    const hiddenPre = new Array(64);
    for (let j = 0; j < 64; j++) {
      hiddenPre[j] = q16Clamp(q16DotTernary(activations, weights.hidden_w[j]) + weights.hidden_b[j]);
    }
    const hidden = hiddenPre.map(q16Relu);

    // Second linear: 64 → 5
    const logits = new Array(5);
    for (let k = 0; k < 5; k++) {
      logits[k] = q16Clamp(q16DotTernary(hidden, weights.output_w[k]) + weights.output_b[k]);
    }

    // Argmax — ties broken by lower index.
    let severity = 0;
    let best = logits[0];
    for (let k = 1; k < 5; k++) {
      if (logits[k] > best) { best = logits[k]; severity = k; }
    }

    // repro_hash = SHA-256 over canonical JSON
    const reproPayload = {
      feature_hash: featureHash,
      logits_q16: logits,
      severity: severity,
      weights_id: weights.bundle_id,
    };
    const reproHash = await sha256Hex(canonicalJson(reproPayload));

    return {
      drug_a: a,
      drug_b: b,
      severity: severity,
      severity_name: SEVERITY_NAMES[severity],
      logits_q16: logits,
      feature_hash: featureHash,
      repro_hash: reproHash,
      weights_id: weights.bundle_id,
    };
  }

  // ─── Self-test (for the iter-69 pin: warfarin + ibuprofen must produce
  // the same repro_hash the Python forward pass produces). The expected
  // hash is captured in tests/test_engine/test_browser_bitnet_pin.py and
  // checked at integration time. ──────────────────────────────────────────

  async function selfTest() {
    const w = await loadWeights();
    const r = await classify("warfarin", "ibuprofen", w);
    return r;
  }

  // ─── Public API ─────────────────────────────────────────────────────────

  window.ClinicalMemBitNet = {
    loadWeights,
    classify,
    encodeDrugToken,
    blake2b,
    selfTest,
    _internals: { canonicalJson, q16DotTernary, q16Clamp, Q16_ONE },
  };

})();
