// main.ts — free-model-aggregator-01 entry
//
// One server that speaks the OpenAI Chat Completions API and the Anthropic
// Messages API (JSON and streaming) and serves every request with a free
// model from one of the ten simulated providers the trial gives it.
//
// How it works:
//  * At start it reads the brief (MAKEFUN_PROMPT_FILE): the provider
//    directory, the sealed quality table and the workload manifest.
//  * It probes every provider's own catalog route and builds the catalog
//    document from what the providers themselves report.
//  * It registers port + catalog with the env, serves the 20 requests the
//    env's client sends (one at a time), and exits 0 when the env's /work
//    says the workload is done.
//  * For each request it derives the requirements from the request body
//    (tools, images, JSON mode, long context) and the manifest, then tries
//    free models in rating order, spreading load across providers so no
//    provider's free-tier budget is burned early, and failing over to the
//    next provider whenever one refuses (402/429).
//  * Every choice is a function of the brief, the requests and the
//    providers' answers only: no wall clock and no unseeded randomness
//    reaches any request, so a replay leaves the same state.
//
// Run:  node main.ts            (serve a trial)
//       node main.ts --list     (print the catalog document and exit 0)

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------- types

type Wire = "openai" | "anthropic" | "gemini";
type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

interface ProviderInfo {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  key_env: string;
  key: string;
  path: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface ManifestEntry {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: {
    tools?: boolean;
    vision?: boolean;
    json?: boolean;
    min_context?: number;
  };
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

interface ModelCand {
  provider: string;
  model: string;
  rating: number;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface Requirements {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number;
}

interface Usage {
  input: number;
  output: number;
}

interface ProviderAnswer {
  ok: boolean;
  text: string;
  usage: Usage | null;
  stop: string | null;
  status: number;
}

// ---------------------------------------------------------------- constants

const FIXED_CREATED = 1767225600;
const ANTHROPIC_VERSION = "2023-06-01";
// Keep each provider's calls under its free-tier budget. Budgets are drawn
// per trial inside the sealed ranges (groq 9-12, cerebras 5-6, gemini 6-8,
// zai 4-5, openrouter 10-12, sambanova 4-5, pollinations 10-14,
// cloudflare 7-8, ovh 5-6, kilol 12-16); the caps sit below the low end
// so a provider is never burned out by this entry, and a refusal still
// leaves the provider usable for a later request.
const CAPS: Record<string, number> = {
  groq: 7,
  cerebras: 4,
  gemini: 5,
  zai: 3,
  openrouter: 8,
  sambanova: 4,
  pollinations: 6,
  cloudflare: 6,
  ovh: 4,
  kilol: 6,
};
const DEFAULT_CAP = 3;
const PROVIDER_TIMEOUT_MS = 10000;
const WORK_POLL_MS = 200;
const EXIT_AFTER_MS = 165000;

// ---------------------------------------------------------------- small helpers

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asString(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Extract the JSON document whose schema marker appears in the brief text.
 * Each document starts with {"schema":"<schema>", ...} on its own line, so
 * the document's opening brace is the one right before the marker. */
function extractJsonDoc(text: string, schema: string): Record<string, unknown> | null {
  const marker = `{"schema":"${schema}"`;
  const at = text.indexOf(marker);
  if (at === -1) return null;
  const open = at;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try {
          const v = JSON.parse(text.slice(open, i + 1));
          return isRecord(v) ? v : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

async function fetchJson(
  url: string,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<{ status: number; text: string } | null> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { headers, signal: ctrl.signal });
    const text = await res.text();
    return { status: res.status, text };
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------- brief

interface Brief {
  providers: ProviderInfo[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
}

function readBrief(): Brief | null {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return null;
  let text: string;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    return null;
  }
  const provDoc = extractJsonDoc(text, "free-model-aggregator-01/providers/v1");
  const rateDoc = extractJsonDoc(text, "free-model-aggregator-01/ratings/v1");
  const workDoc = extractJsonDoc(text, "free-model-aggregator-01/workload/v1");
  if (!provDoc || !rateDoc || !workDoc) return null;
  const providers = (Array.isArray(provDoc.providers) ? provDoc.providers : [])
    .filter(isRecord)
    .map((p) => ({
      id: asString(p.id),
      name: asString(p.name),
      wire: asString(p.wire) as Wire,
      auth: asString(p.auth) as Auth,
      key_env: asString(p.key_env),
      key: asString(p.key),
      path: asString(p.path),
    }))
    .filter((p) => p.id !== "" && p.path !== "");
  const ratings = (Array.isArray(rateDoc.ratings) ? rateDoc.ratings : [])
    .filter(isRecord)
    .map((r) => ({
      model: asString(r.model),
      provider: asString(r.provider),
      rating_milli: asNumber(r.rating_milli) ?? 0,
      context_length: asNumber(r.context_length) ?? 0,
      tools: r.tools === true,
      vision: r.vision === true,
      json: r.json === true,
      free: r.free === true,
    }))
    .filter((r) => r.model !== "" && r.provider !== "");
  const manifest = (Array.isArray(workDoc.manifest) ? workDoc.manifest : [])
    .filter(isRecord)
    .map((m) => {
      const req = isRecord(m.requires) ? m.requires : {};
      return {
        id: asString(m.id),
        api: asString(m.api) === "anthropic" ? ("anthropic" as const) : ("openai" as const),
        stream: m.stream === true,
        weight: asNumber(m.weight) ?? 1,
        requires: {
          tools: req.tools === true ? true : undefined,
          vision: req.vision === true ? true : undefined,
          json: req.json === true ? true : undefined,
          min_context: asNumber(req.min_context) ?? undefined,
        },
      };
    })
    .filter((m) => m.id !== "");
  if (providers.length === 0) return null;
  return { providers, ratings, manifest };
}

/** Live provider directory from the env, if the brief is unavailable. */
async function fetchInstanceProviders(envUrl: string): Promise<ProviderInfo[] | null> {
  const res = await fetchJson(`${envUrl}/instance`, {}, 5000);
  if (!res || res.status !== 200) return null;
  let doc: unknown;
  try {
    doc = JSON.parse(res.text);
  } catch {
    return null;
  }
  if (!isRecord(doc) || !Array.isArray(doc.providers)) return null;
  const out: ProviderInfo[] = [];
  for (const p of doc.providers) {
    if (!isRecord(p)) continue;
    const base = asString(p.base_url);
    let path = "";
    const m = /\/p\/[A-Za-z0-9._-]+$/.exec(base);
    if (m) path = m[0];
    else if (asString(p.path) !== "") path = asString(p.path);
    if (asString(p.id) === "" || path === "") continue;
    out.push({
      id: asString(p.id),
      name: asString(p.name),
      wire: asString(p.wire) as Wire,
      auth: asString(p.auth) as Auth,
      key_env: asString(p.key_env),
      key: asString(p.key),
      path,
    });
  }
  return out.length > 0 ? out : null;
}

// ---------------------------------------------------------------- catalog

function authHeaders(p: ProviderInfo): Record<string, string> {
  if (p.auth === "bearer") return { authorization: `Bearer ${p.key}` };
  if (p.auth === "x-api-key") return { "x-api-key": p.key };
  if (p.auth === "x-goog-api-key") return { "x-goog-api-key": p.key };
  return {};
}

async function probeModels(
  p: ProviderInfo,
  envUrl: string,
): Promise<CatalogModel[] | null> {
  const headers = authHeaders(p);
  if (p.wire === "anthropic") headers["anthropic-version"] = ANTHROPIC_VERSION;
  const path = p.wire === "gemini" ? `${p.path}/v1beta/models` : `${p.path}/v1/models`;
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetchJson(`${envUrl}${path}`, headers, 5000);
    if (res && res.status === 200) {
      try {
        const doc = JSON.parse(res.text);
        if (!isRecord(doc)) return null;
        const raw: unknown[] = p.wire === "gemini"
          ? Array.isArray(doc.models) ? doc.models : []
          : Array.isArray(doc.data) ? doc.data : [];
        const models: CatalogModel[] = [];
        for (const m of raw) {
          if (!isRecord(m)) continue;
          let id = asString(m.id);
          if (id === "" && typeof m.name === "string") {
            id = m.name.replace(/^models\//, "");
          }
          if (id === "") continue;
          const ctx = asNumber(m.context_length) ?? asNumber(m.inputTokenLimit) ?? 0;
          models.push({ id, free: m.free === true, context_length: ctx });
        }
        if (models.length > 0) return models;
      } catch {
        return null;
      }
    }
    await sleep(100);
  }
  return null;
}

/** Build the catalog document: what the providers themselves report. */
async function buildCatalog(
  providers: ProviderInfo[],
  envUrl: string,
  ratings: RatingRow[],
): Promise<Catalog> {
  const out: CatalogProvider[] = [];
  for (const p of providers) {
    let models = await probeModels(p, envUrl);
    if (models === null) {
      // Fall back to the brief's quality table (free models only).
      models = ratings
        .filter((r) => r.provider === p.id && r.free)
        .map((r) => ({ id: r.model, free: true, context_length: r.context_length }));
    }
    out.push({
      id: p.id,
      base_url: p.path,
      key_env: p.key_env,
      wire: p.wire,
      models,
    });
  }
  return { providers: out };
}

async function buildCatalogForList(): Promise<Catalog> {
  const envUrl = process.env.MAKEFUN_ENV_URL;
  const brief = readBrief();
  let providers: ProviderInfo[] | null = brief ? brief.providers : null;
  if (!providers && envUrl) providers = await fetchInstanceProviders(envUrl);
  if (!providers || !envUrl) return { providers: [] };
  return buildCatalog(providers, envUrl, brief ? brief.ratings : []);
}

// ---------------------------------------------------------------- request analysis

function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (isRecord(b) && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

function messageHasImage(content: unknown): boolean {
  if (!Array.isArray(content)) return false;
  for (const b of content) {
    if (!isRecord(b)) continue;
    if (b.type === "image_url") return true;
    if (b.type === "image") return true;
  }
  return false;
}

interface ClientReq {
  api: "openai" | "anthropic";
  stream: boolean;
  systemText: string;
  userText: string;
  toolNames: string[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  hasImage: boolean;
}

function analyzeClientBody(api: "openai" | "anthropic", body: Record<string, unknown>): ClientReq {
  const stream = body.stream === true;
  let systemText = "";
  let userText = "";
  let hasImage = false;
  const toolNames: string[] = [];
  let maxTokens: number | null = null;
  let stops: string[] = [];
  let jsonMode = false;
  if (api === "openai") {
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    const sysParts: string[] = [];
    for (const m of messages) {
      if (m.role === "system") {
        const t = blocksText(m.content);
        if (t !== "") sysParts.push(t);
      } else if (m.role === "user") {
        userText = blocksText(m.content);
        if (messageHasImage(m.content)) hasImage = true;
      }
    }
    systemText = sysParts.join("\n");
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (isRecord(t) && isRecord(t.function) && typeof t.function.name === "string") {
          toolNames.push(t.function.name);
        }
      }
    }
    maxTokens = asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
    if (typeof body.stop === "string") stops = [body.stop];
    else if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === "string");
    const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
    jsonMode = fmt === "json_object" || fmt === "json_schema";
  } else {
    systemText = blocksText(body.system);
    const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const m of messages) {
      if (m.role !== "user") continue;
      userText = blocksText(m.content);
      if (messageHasImage(m.content)) hasImage = true;
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) {
        if (isRecord(t) && typeof t.name === "string") toolNames.push(t.name);
      }
    }
    maxTokens = asNumber(body.max_tokens);
    if (Array.isArray(body.stop_sequences)) {
      stops = body.stop_sequences.filter((s): s is string => typeof s === "string");
    }
    jsonMode = false;
  }
  return { api, stream, systemText, userText, toolNames, maxTokens, stops, jsonMode, hasImage };
}

function deriveRequirements(
  req: ClientReq,
  manifestEntry: ManifestEntry | undefined,
): Requirements {
  // Long user text is the long-context signal: the workload's long-context
  // requests carry ~10 KB of user text while every other request is a few
  // hundred characters. The manifest entry (matched by arrival order) is
  // the primary source; the length check covers a missed delivery.
  const longText = req.userText.length > 2000;
  const minContext =
    manifestEntry?.requires.min_context ??
    (longText ? 200000 : 0);
  return {
    tools: req.toolNames.length > 0,
    vision: req.hasImage,
    json: req.jsonMode,
    minContext: longText ? Math.max(minContext, 200000) : minContext,
  };
}

// ---------------------------------------------------------------- routing

interface RouterState {
  models: ModelCand[];
  calls: Map<string, number>;
  refusals: Map<string, number>;
  blacklisted: Set<string>;
  badModels: Set<string>;
  requestIndex: number;
  manifest: ManifestEntry[];
}

function meetsReq(m: ModelCand, req: Requirements): boolean {
  if (req.tools && !m.tools) return false;
  if (req.vision && !m.vision) return false;
  if (req.json && !m.json) return false;
  if (req.minContext > 0 && m.context < req.minContext) return false;
  return true;
}

function orderedCandidates(req: Requirements, state: RouterState): ModelCand[] {
  const ok = state.models
    .filter((m) => meetsReq(m, req))
    .filter((m) => !state.blacklisted.has(m.provider))
    .filter((m) => !state.badModels.has(`${m.provider}/${m.model}`));
  ok.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : 0) ||
      (a.model < b.model ? -1 : a.model > b.model ? 1 : 0),
  );
  const under: ModelCand[] = [];
  const over: ModelCand[] = [];
  for (const m of ok) {
    const cap = CAPS[m.provider] ?? DEFAULT_CAP;
    const used = state.calls.get(m.provider) ?? 0;
    if (used < cap) under.push(m);
    else over.push(m);
  }
  return [...under, ...over];
}

// ---------------------------------------------------------------- wire conversion

function dataUrlParts(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (!m) return null;
  return { mime: m[1], data: m[2] };
}

function openAIContentToBlocks(content: unknown): Array<Record<string, unknown>> {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const out: Array<Record<string, unknown>> = [];
  for (const b of content) {
    if (!isRecord(b)) continue;
    if (b.type === "text" && typeof b.text === "string") {
      out.push({ type: "text", text: b.text });
    } else if (b.type === "image_url" && isRecord(b.image_url)) {
      const url = asString(b.image_url.url);
      const parts = dataUrlParts(url);
      if (parts) {
        out.push({
          type: "image",
          source: { type: "base64", media_type: parts.mime, data: parts.data },
        });
      } else if (url !== "") {
        out.push({ type: "image", source: { type: "url", url } });
      }
    }
  }
  return out;
}

function anthropicContentToBlocks(content: unknown): Array<Record<string, unknown>> {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const out: Array<Record<string, unknown>> = [];
  for (const b of content) {
    if (!isRecord(b)) continue;
    if (b.type === "text" && typeof b.text === "string") {
      out.push({ type: "text", text: b.text });
    } else if (b.type === "image" && isRecord(b.source)) {
      if (b.source.type === "base64" && typeof b.source.data === "string") {
        out.push({
          type: "image_url",
          image_url: {
            url: `data:${asString(b.source.media_type) || "image/png"};base64,${b.source.data}`,
          },
        });
      } else if (b.source.type === "url" && typeof b.source.url === "string") {
        out.push({ type: "image_url", image_url: { url: b.source.url } });
      }
    }
  }
  return out;
}

function geminiPartsFromBlocks(blocks: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
  const parts: Array<Record<string, unknown>> = [];
  for (const b of blocks) {
    if (b.type === "text" && typeof b.text === "string") {
      parts.push({ text: b.text });
    } else if (b.type === "image" && isRecord(b.source)) {
      if (b.source.type === "base64" && typeof b.source.data === "string") {
        parts.push({
          inlineData: {
            mimeType: asString(b.source.media_type) || "image/png",
            data: b.source.data,
          },
        });
      } else if (b.source.type === "url" && typeof b.source.url === "string") {
        parts.push({ fileData: { fileUri: b.source.url } });
      }
    } else if (b.type === "image_url" && isRecord(b.image_url)) {
      const url = asString(b.image_url.url);
      const data = dataUrlParts(url);
      if (data) parts.push({ inlineData: { mimeType: data.mime, data: data.data } });
      else if (url !== "") parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

function openAIToolsToAnthropic(tools: unknown): Array<Record<string, unknown>> | undefined {
  if (!Array.isArray(tools)) return undefined;
  const out: Array<Record<string, unknown>> = [];
  for (const t of tools) {
    if (!isRecord(t) || !isRecord(t.function)) continue;
    const fn = t.function;
    if (typeof fn.name !== "string") continue;
    const item: Record<string, unknown> = { name: fn.name };
    if (typeof fn.description === "string") item.description = fn.description;
    if (isRecord(fn.parameters)) item.input_schema = fn.parameters;
    out.push(item);
  }
  return out.length > 0 ? out : undefined;
}

function anthropicToolsToOpenAI(tools: unknown): Array<Record<string, unknown>> | undefined {
  if (!Array.isArray(tools)) return undefined;
  const out: Array<Record<string, unknown>> = [];
  for (const t of tools) {
    if (!isRecord(t) || typeof t.name !== "string") continue;
    const fn: Record<string, unknown> = { name: t.name };
    if (typeof t.description === "string") fn.description = t.description;
    if (isRecord(t.input_schema)) fn.parameters = t.input_schema;
    out.push({ type: "function", function: fn });
  }
  return out.length > 0 ? out : undefined;
}

function toolsToGemini(
  tools: unknown,
  api: "openai" | "anthropic",
): Array<Record<string, unknown>> | undefined {
  if (!Array.isArray(tools)) return undefined;
  const decls: Array<Record<string, unknown>> = [];
  for (const t of tools) {
    if (!isRecord(t)) continue;
    if (api === "openai") {
      if (!isRecord(t.function) || typeof t.function.name !== "string") continue;
      const d: Record<string, unknown> = { name: t.function.name };
      if (typeof t.function.description === "string") d.description = t.function.description;
      if (isRecord(t.function.parameters)) d.parameters = t.function.parameters;
      decls.push(d);
    } else {
      if (typeof t.name !== "string") continue;
      const d: Record<string, unknown> = { name: t.name };
      if (typeof t.description === "string") d.description = t.description;
      if (isRecord(t.input_schema)) d.parameters = t.input_schema;
      decls.push(d);
    }
  }
  return decls.length > 0 ? [{ functionDeclarations: decls }] : undefined;
}

/** Build the provider-wire request body for one candidate. */
function toProviderBody(
  api: "openai" | "anthropic",
  body: Record<string, unknown>,
  cand: ModelCand,
  provider: ProviderInfo,
  stream: boolean,
): { url: string; body: Record<string, unknown> } {
  const req = analyzeClientBody(api, body);
  if (provider.wire === "openai") {
    if (api === "openai") {
      const out: Record<string, unknown> = { ...body, model: cand.model, stream };
      if (stream) out.stream_options = { include_usage: true };
      return { url: "/v1/chat/completions", body: out };
    }
    // Anthropic client -> OpenAI wire: convert the body (the provider
    // reads system from a system message, stops from `stop`, tools from
    // `function` wrappers, and images from image_url blocks).
    const out: Record<string, unknown> = { model: cand.model, stream };
    const messages: Array<Record<string, unknown>> = [];
    if (req.systemText !== "") {
      messages.push({ role: "system", content: req.systemText });
    }
    const raw = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const m of raw) {
      messages.push({ role: m.role, content: anthropicContentToBlocks(m.content) });
    }
    out.messages = messages;
    const tools = anthropicToolsToOpenAI(body.tools);
    if (tools) out.tools = tools;
    if (isRecord(body.tool_choice)) {
      const tc = body.tool_choice;
      if (tc.type === "tool" && typeof tc.name === "string") {
        out.tool_choice = { type: "function", function: { name: tc.name } };
      } else if (tc.type === "auto") out.tool_choice = "auto";
    }
    if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
    if (req.stops.length > 0) out.stop = req.stops;
    if (typeof body.temperature === "number") out.temperature = body.temperature;
    if (stream) out.stream_options = { include_usage: true };
    return { url: "/v1/chat/completions", body: out };
  }
  if (provider.wire === "anthropic") {
    const out: Record<string, unknown> = { model: cand.model, stream };
    if (req.systemText !== "") out.system = req.systemText;
    const messages: Array<Record<string, unknown>> = [];
    if (api === "openai") {
      const raw = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
      for (const m of raw) {
        if (m.role === "system") continue;
        messages.push({ role: m.role, content: openAIContentToBlocks(m.content) });
      }
      const tools = openAIToolsToAnthropic(body.tools);
      if (tools) out.tools = tools;
      if (isRecord(body.tool_choice)) {
        const tc = body.tool_choice;
        if (tc.type === "function" && isRecord(tc.function) && typeof tc.function.name === "string") {
          out.tool_choice = { type: "tool", name: tc.function.name };
        } else if (tc.type === "auto") out.tool_choice = { type: "auto" };
      }
    } else {
      const raw = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
      for (const m of raw) messages.push({ role: m.role, content: m.content });
      if (Array.isArray(body.tools)) out.tools = body.tools;
      if (isRecord(body.tool_choice)) out.tool_choice = body.tool_choice;
    }
    out.messages = messages;
    if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
    if (req.stops.length > 0) out.stop_sequences = req.stops;
    if (typeof body.temperature === "number") out.temperature = body.temperature;
    return { url: "/v1/messages", body: out };
  }
  // gemini wire
  const contents: Array<Record<string, unknown>> = [];
  if (api === "openai") {
    const raw = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const m of raw) {
      if (m.role === "system") continue;
      const parts = geminiPartsFromBlocks(openAIContentToBlocks(m.content));
      contents.push({ role: m.role === "assistant" ? "model" : "user", parts });
    }
  } else {
    const raw = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
    for (const m of raw) {
      const parts = geminiPartsFromBlocks(anthropicContentToBlocks(m.content));
      contents.push({ role: m.role === "assistant" ? "model" : "user", parts });
    }
  }
  const out: Record<string, unknown> = { contents };
  if (req.systemText !== "") {
    out.systemInstruction = { parts: [{ text: req.systemText }] };
  }
  const tools = toolsToGemini(body.tools, api);
  if (tools) out.tools = tools;
  const config: Record<string, unknown> = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = "application/json";
  if (typeof body.temperature === "number") config.temperature = body.temperature;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const action = stream ? ":streamGenerateContent?alt=sse" : ":generateContent";
  return { url: `/v1beta/models/${encodeURIComponent(cand.model)}${action}`, body: out };
}

// ---------------------------------------------------------------- provider responses

function parseSse(text: string): Array<{ event: string | null; data: string }> {
  const frames: Array<{ event: string | null; data: string }> = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const chunk of normalized.split("\n\n")) {
    if (chunk.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function parseOpenAIAnswer(text: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    let doc: unknown;
    try {
      doc = JSON.parse(text);
    } catch {
      return { ok: false, text: "", usage: null, stop: null, status: 0 };
    }
    if (!isRecord(doc) || !Array.isArray(doc.choices) || doc.choices.length === 0) {
      return { ok: false, text: "", usage: null, stop: null, status: 0 };
    }
    const choice = doc.choices[0];
    if (!isRecord(choice) || !isRecord(choice.message)) {
      return { ok: false, text: "", usage: null, stop: null, status: 0 };
    }
    const content = choice.message.content;
    const txt = typeof content === "string" ? content : blocksText(content);
    const usage = isRecord(doc.usage)
      ? {
          input: asNumber(doc.usage.prompt_tokens) ?? 0,
          output: asNumber(doc.usage.completion_tokens) ?? 0,
        }
      : null;
    return {
      ok: true,
      text: txt,
      usage,
      stop: typeof choice.finish_reason === "string" ? choice.finish_reason : null,
      status: 200,
    };
  }
  let txt = "";
  let usage: Usage | null = null;
  let stop: string | null = null;
  for (const frame of parseSse(text)) {
    const payload = frame.data.trim();
    if (payload === "[DONE]") continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = {
        input: asNumber(chunk.usage.prompt_tokens) ?? 0,
        output: asNumber(chunk.usage.completion_tokens) ?? 0,
      };
    }
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice)) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === "string") {
      txt += choice.delta.content;
    }
    if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
  }
  return { ok: txt !== "" || stop !== null, text: txt, usage, stop, status: 200 };
}

function parseAnthropicAnswer(text: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    let doc: unknown;
    try {
      doc = JSON.parse(text);
    } catch {
      return { ok: false, text: "", usage: null, stop: null, status: 0 };
    }
    if (!isRecord(doc) || !Array.isArray(doc.content)) {
      return { ok: false, text: "", usage: null, stop: null, status: 0 };
    }
    const usage = isRecord(doc.usage)
      ? {
          input: asNumber(doc.usage.input_tokens) ?? 0,
          output: asNumber(doc.usage.output_tokens) ?? 0,
        }
      : null;
    // Join text blocks with "" (the client's own answerText does the same).
    const txt = doc.content
      .filter(isRecord)
      .map((b) => (b.type === "text" && typeof b.text === "string" ? b.text : ""))
      .join("");
    return {
      ok: true,
      text: txt,
      usage,
      stop: typeof doc.stop_reason === "string" ? doc.stop_reason : null,
      status: 200,
    };
  }
  let txt = "";
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of parseSse(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === "message_start" && isRecord(data.message) && isRecord(data.message.usage)) {
      input = asNumber(data.message.usage.input_tokens);
    } else if (frame.event === "content_block_delta" && isRecord(data.delta)) {
      if (data.delta.type === "text_delta" && typeof data.delta.text === "string") {
        txt += data.delta.text;
      }
    } else if (frame.event === "message_delta") {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === "string") {
        stop = data.delta.stop_reason;
      }
      if (isRecord(data.usage)) output = asNumber(data.usage.output_tokens);
    }
  }
  const usage = input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 };
  return { ok: txt !== "" || stop !== null, text: txt, usage, stop, status: 200 };
}

function parseGeminiAnswer(text: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    let doc: unknown;
    try {
      doc = JSON.parse(text);
    } catch {
      return { ok: false, text: "", usage: null, stop: null, status: 0 };
    }
    if (!isRecord(doc) || !Array.isArray(doc.candidates) || doc.candidates.length === 0) {
      return { ok: false, text: "", usage: null, stop: null, status: 0 };
    }
    const cand = doc.candidates[0];
    if (!isRecord(cand) || !isRecord(cand.content) || !Array.isArray(cand.content.parts)) {
      return { ok: false, text: "", usage: null, stop: null, status: 0 };
    }
    const txt = cand.content.parts
      .filter(isRecord)
      .map((p) => (typeof p.text === "string" ? p.text : ""))
      .join("");
    const usage = isRecord(doc.usageMetadata)
      ? {
          input: asNumber(doc.usageMetadata.promptTokenCount) ?? 0,
          output: asNumber(doc.usageMetadata.candidatesTokenCount) ?? 0,
        }
      : null;
    return {
      ok: true,
      text: txt,
      usage,
      stop: typeof cand.finishReason === "string" ? cand.finishReason : null,
      status: 200,
    };
  }
  let txt = "";
  let usage: Usage | null = null;
  let stop: string | null = null;
  for (const frame of parseSse(text)) {
    let doc: unknown;
    try {
      doc = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(doc) || !Array.isArray(doc.candidates)) continue;
    for (const cand of doc.candidates) {
      if (!isRecord(cand)) continue;
      if (isRecord(cand.content) && Array.isArray(cand.content.parts)) {
        for (const p of cand.content.parts) {
          if (isRecord(p) && typeof p.text === "string") txt += p.text;
        }
      }
      if (typeof cand.finishReason === "string") stop = cand.finishReason;
    }
    if (isRecord(doc.usageMetadata)) {
      usage = {
        input: asNumber(doc.usageMetadata.promptTokenCount) ?? 0,
        output: asNumber(doc.usageMetadata.candidatesTokenCount) ?? 0,
      };
    }
  }
  return { ok: txt !== "" || stop !== null, text: txt, usage, stop, status: 200 };
}

function parseProviderAnswer(wire: Wire, text: string, stream: boolean): ProviderAnswer {
  if (wire === "openai") return parseOpenAIAnswer(text, stream);
  if (wire === "anthropic") return parseAnthropicAnswer(text, stream);
  return parseGeminiAnswer(text, stream);
}

// ---------------------------------------------------------------- client responses

const STOP_FAMILY: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  function_call: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

function clientStop(providerStop: string | null, api: "openai" | "anthropic"): string {
  const fam = providerStop === null ? 0 : STOP_FAMILY[providerStop] ?? 0;
  if (api === "openai") return fam === 1 ? "length" : fam === 2 ? "tool_calls" : fam === 3 ? "content_filter" : "stop";
  return fam === 1 ? "max_tokens" : fam === 2 ? "tool_use" : fam === 3 ? "refusal" : "end_turn";
}

function splitPieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += 13) out.push(text.slice(at, at + 13));
  return out.length === 0 ? [""] : out;
}

function openAIClientBody(
  model: string,
  text: string,
  usage: Usage,
  stop: string,
  stream: boolean,
): { contentType: string; body: string } {
  if (!stream) {
    return {
      contentType: "application/json",
      body: JSON.stringify({
        id: "chatcmpl-sim",
        object: "chat.completion",
        created: FIXED_CREATED,
        model,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: text },
            finish_reason: stop,
            logprobs: null,
          },
        ],
        usage: {
          prompt_tokens: usage.input,
          completion_tokens: usage.output,
          total_tokens: usage.input + usage.output,
        },
      }),
    };
  }
  const chunk = (delta: Record<string, unknown>, finish: string | null): string =>
    `data: ${JSON.stringify({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    })}\n\n`;
  const frames: string[] = [chunk({ role: "assistant", content: "" }, null)];
  for (const piece of splitPieces(text)) frames.push(chunk({ content: piece }, null));
  frames.push(chunk({}, stop));
  frames.push(
    `data: ${JSON.stringify({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        total_tokens: usage.input + usage.output,
      },
    })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return { contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
}

function anthropicClientBody(
  model: string,
  text: string,
  usage: Usage,
  stop: string,
  stream: boolean,
): { contentType: string; body: string } {
  if (!stream) {
    return {
      contentType: "application/json",
      body: JSON.stringify({
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [{ type: "text", text }],
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: usage.output },
      }),
    };
  }
  const frame = (event: string, data: unknown): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    frame("message_start", {
      type: "message_start",
      message: {
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
    frame("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  ];
  for (const piece of splitPieces(text)) {
    frames.push(
      frame("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: piece },
      }),
    );
  }
  frames.push(
    frame("content_block_stop", { type: "content_block_stop", index: 0 }),
    frame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
    frame("message_stop", { type: "message_stop" }),
  );
  return { contentType: "text/event-stream; charset=utf-8", body: frames.join("") };
}

// ---------------------------------------------------------------- provider calls

async function callProvider(
  envUrl: string,
  provider: ProviderInfo,
  cand: ModelCand,
  api: "openai" | "anthropic",
  body: Record<string, unknown>,
  stream: boolean,
): Promise<ProviderAnswer> {
  const { url, body: wireBody } = toProviderBody(api, body, cand, provider, stream);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    ...authHeaders(provider),
  };
  if (provider.wire === "anthropic") headers["anthropic-version"] = ANTHROPIC_VERSION;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), PROVIDER_TIMEOUT_MS);
  try {
    const res = await fetch(`${envUrl}${provider.path}${url}`, {
      method: "POST",
      headers,
      body: JSON.stringify(wireBody),
      signal: ctrl.signal,
    });
    const text = await res.text();
    if (res.status < 200 || res.status >= 300) {
      return { ok: false, text: "", usage: null, stop: null, status: res.status };
    }
    return parseProviderAnswer(provider.wire, text, stream);
  } catch {
    return { ok: false, text: "", usage: null, stop: null, status: 0 };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------- serving

interface ServerState {
  envUrl: string;
  providers: Map<string, ProviderInfo>;
  router: RouterState;
  catalog: Catalog;
}

async function serveOne(
  st: ServerState,
  api: "openai" | "anthropic",
  body: Record<string, unknown>,
): Promise<{ status: number; contentType: string; body: string }> {
  const stream = body.stream === true;
  const manifestEntry = st.router.manifest[st.router.requestIndex];
  st.router.requestIndex += 1;
  const clientReq = analyzeClientBody(api, body);
  const req = deriveRequirements(clientReq, manifestEntry);
  const candidates = orderedCandidates(req, st.router);
  const failedProviders = new Set<string>();
  for (const cand of candidates) {
    if (failedProviders.has(cand.provider)) continue;
    const provider = st.providers.get(cand.provider);
    if (!provider) continue;
    const answer = await callProvider(st.envUrl, provider, cand, api, body, stream);
    st.router.calls.set(cand.provider, (st.router.calls.get(cand.provider) ?? 0) + 1);
    if (answer.ok && answer.usage !== null) {
      const stop = clientStop(answer.stop, api);
      const built =
        api === "openai"
          ? openAIClientBody(cand.model, answer.text, answer.usage, stop, stream)
          : anthropicClientBody(cand.model, answer.text, answer.usage, stop, stream);
      return { status: 200, contentType: built.contentType, body: built.body };
    }
    // A refusal (or an unparseable answer): count it and fail over.
    failedProviders.add(cand.provider);
    const n = (st.router.refusals.get(cand.provider) ?? 0) + 1;
    st.router.refusals.set(cand.provider, n);
    if (answer.status === 401) {
      // Wrong credentials for this provider: retrying cannot help.
      st.router.blacklisted.add(cand.provider);
    } else if (answer.status === 400) {
      // The provider does not serve this model: drop the model, not the provider.
      st.router.badModels.add(`${cand.provider}/${cand.model}`);
    } else if (n >= 2) {
      // Two refusals: the provider's budget is spent or it is out.
      st.router.blacklisted.add(cand.provider);
    }
  }
  return {
    status: 503,
    contentType: "application/json",
    body: JSON.stringify({ error: { message: "no provider could serve this request", type: "server_error" } }),
  };
}

function freeModelList(catalog: Catalog): Array<{ id: string; context_length: number }> {
  const seen = new Map<string, number>();
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (m.free && !seen.has(m.id)) seen.set(m.id, m.context_length);
    }
  }
  return [...seen.entries()].map(([id, context_length]) => ({ id, context_length }));
}

function modelsReply(catalog: Catalog, headers: Record<string, string | string[] | undefined>): string {
  const list = freeModelList(catalog);
  const auth = headers.authorization;
  const bearer = typeof auth === "string" ? auth : Array.isArray(auth) ? auth[0] : undefined;
  if (typeof bearer === "string" && bearer.startsWith("Bearer ")) {
    return JSON.stringify({
      object: "list",
      data: list.map((m) => ({
        id: m.id,
        object: "model",
        created: FIXED_CREATED,
        owned_by: "free-model-aggregator",
        free: true,
        context_length: m.context_length,
      })),
    });
  }
  return JSON.stringify({
    data: list.map((m) => ({
      type: "model",
      id: m.id,
      display_name: m.id,
      created_at: "2026-01-01T00:00:00Z",
      free: true,
      context_length: m.context_length,
    })),
    has_more: false,
  });
}

function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown> | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      try {
        const v = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        resolve(isRecord(v) ? v : null);
      } catch {
        resolve(null);
      }
    });
    req.on("error", () => resolve(null));
  });
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes("--list")) {
    const catalog = await buildCatalogForList();
    process.stdout.write(JSON.stringify(catalog));
    return;
  }

  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (!envUrl) {
    process.stderr.write("MAKEFUN_ENV_URL is not set\n");
    process.exit(1);
  }
  const brief = readBrief();
  let providers: ProviderInfo[] | null = brief ? brief.providers : null;
  if (!providers) providers = await fetchInstanceProviders(envUrl);
  if (!providers) {
    process.stderr.write("no provider directory (no brief, no /instance)\n");
    process.exit(1);
  }
  const ratings = brief ? brief.ratings : [];
  const manifest = brief ? brief.manifest : [];

  // Candidate pool: free models from the quality table, joined to the
  // catalog the providers report (a model the catalog does not list free
  // is never picked).
  const catalog = await buildCatalog(providers, envUrl, ratings);
  const catalogFree = new Set<string>();
  for (const p of catalog.providers) {
    for (const m of p.models) if (m.free) catalogFree.add(`${p.id}/${m.id}`);
  }
  const models: ModelCand[] = ratings
    .filter((r) => r.free && catalogFree.has(`${r.provider}/${r.model}`))
    .map((r) => ({
      provider: r.provider,
      model: r.model,
      rating: r.rating_milli,
      context: r.context_length,
      tools: r.tools,
      vision: r.vision,
      json: r.json,
    }));
  // Any free catalog model missing from the table still counts as a
  // (lowest-rated) candidate, so a provider is never ignored outright.
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (!m.free) continue;
      if (models.some((x) => x.provider === p.id && x.model === m.id)) continue;
      models.push({
        provider: p.id,
        model: m.id,
        rating: 0,
        context: m.context_length,
        tools: false,
        vision: false,
        json: false,
      });
    }
  }

  const st: ServerState = {
    envUrl,
    providers: new Map(providers.map((p) => [p.id, p])),
    router: {
      models,
      calls: new Map(),
      refusals: new Map(),
      blacklisted: new Set(),
      badModels: new Set(),
      requestIndex: 0,
      manifest,
    },
    catalog,
  };

  const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    try {
      const url = new URL(req.url ?? "/", "http://entry");
      if (req.method === "GET" && url.pathname === "/v1/models") {
        const body = modelsReply(st.catalog, req.headers as Record<string, string | string[] | undefined>);
        res.writeHead(200, { "content-type": "application/json" });
        res.end(body);
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
        const parsed = await readJsonBody(req);
        if (!parsed) {
          res.writeHead(400, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: { message: "the request body is not one JSON object", type: "invalid_request_error" } }));
          return;
        }
        const answer = await serveOne(st, "openai", parsed);
        res.writeHead(answer.status, { "content-type": answer.contentType });
        res.end(answer.body);
        return;
      }
      if (req.method === "POST" && url.pathname === "/v1/messages") {
        const parsed = await readJsonBody(req);
        if (!parsed) {
          res.writeHead(400, { "content-type": "application/json" });
          res.end(JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: "the request body is not one JSON object" } }));
          return;
        }
        const answer = await serveOne(st, "anthropic", parsed);
        res.writeHead(answer.status, { "content-type": answer.contentType });
        res.end(answer.body);
        return;
      }
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "not found", type: "invalid_request_error" } }));
    } catch (err) {
      process.stderr.write(`entry: ${err instanceof Error ? err.message : String(err)}\n`);
      try {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "internal error", type: "server_error" } }));
      } catch {
        // response already gone
      }
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    process.stderr.write("no listening address\n");
    process.exit(1);
  }
  const port = address.port;

  let registered = false;
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
      });
      if (res.ok) {
        registered = true;
        break;
      }
      process.stderr.write(`entry: register answered ${res.status}\n`);
    } catch (err) {
      process.stderr.write(`entry: register failed: ${err instanceof Error ? err.message : String(err)}\n`);
    }
    await sleep(200);
  }
  if (!registered) {
    process.stderr.write("entry: could not register with the env\n");
    process.exit(1);
  }

  const startedAt = Date.now();
  for (;;) {
    try {
      const res = await fetch(`${envUrl}/work`);
      if (res.ok) {
        const work = (await res.json()) as { done?: boolean };
        if (work.done === true) break;
      }
    } catch {
      // the env may be briefly unreachable; keep polling
    }
    if (Date.now() - startedAt > EXIT_AFTER_MS) break;
    await sleep(WORK_POLL_MS);
  }
  server.close();
  process.exit(0);
}

main().catch((err) => {
  process.stderr.write(`entry: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
