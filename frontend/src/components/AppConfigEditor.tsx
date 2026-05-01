import { useEffect, useState } from "react";
import { AppConfig, getApp, registerApp, updateApp } from "../api/client";

const DEFAULT_CONFIG: AppConfig = {
  app_name: "",
  collection: "",
  ingestion: {
    strategy: "semantic_doc",
    max_tokens: 512,
    min_tokens: 50,
    overlap_tokens: 64,
    semantic_split: true,
  },
  embedding: {
    model: "BAAI/bge-small-en",
    batch_size: 32,
    normalize: true,
  },
  retrieval: {
    top_k: 10,
    hybrid: true,
    alpha: 0.7,
  },
  reranking: {
    enabled: true,
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: 5,
    candidate_cap: 20,
  },
  generation: {
    response_type: "markdown",
    temperature: 0.1,
    strict: false,
    max_retries: 2,
  },
  defaults: {
    system_prompt: "",
  },
  tasks: {},
  default_task: null,
};

interface Props {
  initialAppName: string | null;
  onSaved: () => void;
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400">{label}</label>
      {children}
    </div>
  );
}

function Input({
  value,
  onChange,
  type = "text",
  disabled,
}: {
  value: string | number;
  onChange: (v: string) => void;
  type?: string;
  disabled?: boolean;
}) {
  return (
    <input
      type={type}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none disabled:opacity-50"
    />
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function Toggle({
  value,
  onChange,
  label,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer select-none">
      <div
        onClick={() => onChange(!value)}
        className={`w-8 h-4 rounded-full transition-colors relative ${
          value ? "bg-indigo-600" : "bg-gray-700"
        }`}
      >
        <div
          className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
            value ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </div>
      <span className="text-xs text-gray-300">{label}</span>
    </label>
  );
}

function SectionHeader({ title }: { title: string }) {
  return (
    <h3 className="border-b border-gray-800 pb-1 text-xs font-semibold text-indigo-400 uppercase tracking-widest">
      {title}
    </h3>
  );
}

export default function AppConfigEditor({ initialAppName, onSaved }: Props) {
  const [config, setConfig] = useState<AppConfig>(DEFAULT_CONFIG);
  const [rawJson, setRawJson] = useState("");
  const [jsonMode, setJsonMode] = useState(false);
  const [isEdit, setIsEdit] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  useEffect(() => {
    if (initialAppName) {
      setIsEdit(true);
      getApp(initialAppName).then((cfg) => {
        setConfig(cfg);
        setRawJson(JSON.stringify(cfg, null, 2));
      });
    } else {
      setIsEdit(false);
      setConfig(DEFAULT_CONFIG);
      setRawJson(JSON.stringify(DEFAULT_CONFIG, null, 2));
    }
  }, [initialAppName]);

  function set<T>(path: string[], value: T) {
    setConfig((prev) => {
      const next = structuredClone(prev) as Record<string, unknown>;
      let cur = next;
      for (let i = 0; i < path.length - 1; i++) {
        cur = cur[path[i]] as Record<string, unknown>;
      }
      cur[path[path.length - 1]] = value;
      return next as AppConfig;
    });
  }

  function validate(): string[] {
    const errs: string[] = [];
    if (!config.app_name.trim()) errs.push("app_name is required");
    if (!config.collection.trim()) errs.push("collection is required");
    if (!/^[a-z][a-z0-9_-]{0,62}$/.test(config.app_name))
      errs.push("app_name: lowercase letters, digits, hyphens, underscores only");
    if (!/^[a-z][a-z0-9_-]{0,62}$/.test(config.collection))
      errs.push("collection: lowercase letters, digits, hyphens, underscores only");
    const d = config.defaults as Record<string, unknown>;
    if (!String(d.system_prompt ?? "").trim())
      errs.push("defaults.system_prompt is required");
    const ing = config.ingestion as Record<string, unknown>;
    if (Number(ing.min_tokens) >= Number(ing.max_tokens))
      errs.push("ingestion: min_tokens must be less than max_tokens");
    return errs;
  }

  async function handleSave() {
    if (jsonMode) {
      try {
        const parsed = JSON.parse(rawJson);
        setConfig(parsed);
      } catch {
        setError("Invalid JSON");
        return;
      }
    }

    const errs = validate();
    if (errs.length > 0) {
      setValidationErrors(errs);
      return;
    }

    setValidationErrors([]);
    setSaving(true);
    setError(null);
    try {
      if (isEdit) {
        const { app_name, ...rest } = config;
        await updateApp(app_name, rest);
      } else {
        await registerApp(config);
      }
      onSaved();
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "response" in e
          ? JSON.stringify((e as { response: { data: unknown } }).response.data, null, 2)
          : String(e);
      setError(msg);
    } finally {
      setSaving(false);
    }
  }

  const ing = config.ingestion as Record<string, unknown>;
  const emb = config.embedding as Record<string, unknown>;
  const ret = config.retrieval as Record<string, unknown>;
  const rer = config.reranking as Record<string, unknown>;
  const gen = config.generation as Record<string, unknown>;
  const def = config.defaults as Record<string, unknown>;

  return (
    <div className="flex flex-col gap-5 max-w-3xl">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-200">
          {isEdit ? `Edit: ${config.app_name}` : "New Application"}
        </h2>
        <button
          onClick={() => setJsonMode(!jsonMode)}
          className="text-xs text-gray-500 hover:text-gray-300 underline"
        >
          {jsonMode ? "Form view" : "JSON view"}
        </button>
      </div>

      {validationErrors.length > 0 && (
        <div className="rounded border border-yellow-800 bg-yellow-950 px-3 py-2 text-xs text-yellow-300">
          <ul className="list-disc list-inside space-y-0.5">
            {validationErrors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <div className="rounded border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-300 whitespace-pre-wrap">
          {error}
        </div>
      )}

      {jsonMode ? (
        <textarea
          value={rawJson}
          onChange={(e) => setRawJson(e.target.value)}
          rows={28}
          className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-xs font-mono text-gray-100 focus:border-indigo-500 focus:outline-none"
        />
      ) : (
        <div className="flex flex-col gap-4">
          {/* Identity */}
          <SectionHeader title="Identity" />
          <div className="grid grid-cols-2 gap-3">
            <Field label="app_name">
              <Input
                value={config.app_name}
                onChange={(v) => set(["app_name"], v)}
                disabled={isEdit}
              />
            </Field>
            <Field label="collection">
              <Input
                value={config.collection}
                onChange={(v) => set(["collection"], v)}
              />
            </Field>
          </div>

          {/* Ingestion */}
          <SectionHeader title="Ingestion" />
          <div className="grid grid-cols-3 gap-3">
            <Field label="strategy">
              <Select
                value={String(ing.strategy)}
                onChange={(v) => set(["ingestion", "strategy"], v)}
                options={["semantic_doc", "resume_structured", "markdown_aware"]}
              />
            </Field>
            <Field label="max_tokens">
              <Input
                type="number"
                value={Number(ing.max_tokens)}
                onChange={(v) => set(["ingestion", "max_tokens"], Number(v))}
              />
            </Field>
            <Field label="min_tokens">
              <Input
                type="number"
                value={Number(ing.min_tokens)}
                onChange={(v) => set(["ingestion", "min_tokens"], Number(v))}
              />
            </Field>
            <Field label="overlap_tokens">
              <Input
                type="number"
                value={Number(ing.overlap_tokens)}
                onChange={(v) => set(["ingestion", "overlap_tokens"], Number(v))}
              />
            </Field>
            <div className="flex items-end pb-1">
              <Toggle
                value={Boolean(ing.semantic_split)}
                onChange={(v) => set(["ingestion", "semantic_split"], v)}
                label="semantic_split"
              />
            </div>
          </div>

          {/* Embedding */}
          <SectionHeader title="Embedding" />
          <div className="grid grid-cols-3 gap-3">
            <Field label="model">
              <Input
                value={String(emb.model)}
                onChange={(v) => set(["embedding", "model"], v)}
              />
            </Field>
            <Field label="batch_size">
              <Input
                type="number"
                value={Number(emb.batch_size)}
                onChange={(v) => set(["embedding", "batch_size"], Number(v))}
              />
            </Field>
            <div className="flex items-end pb-1">
              <Toggle
                value={Boolean(emb.normalize)}
                onChange={(v) => set(["embedding", "normalize"], v)}
                label="normalize"
              />
            </div>
          </div>

          {/* Retrieval */}
          <SectionHeader title="Retrieval" />
          <div className="grid grid-cols-3 gap-3">
            <Field label="top_k">
              <Input
                type="number"
                value={Number(ret.top_k)}
                onChange={(v) => set(["retrieval", "top_k"], Number(v))}
              />
            </Field>
            <Field label="alpha (dense weight)">
              <Input
                type="number"
                value={Number(ret.alpha)}
                onChange={(v) => set(["retrieval", "alpha"], Number(v))}
              />
            </Field>
            <div className="flex items-end pb-1">
              <Toggle
                value={Boolean(ret.hybrid)}
                onChange={(v) => set(["retrieval", "hybrid"], v)}
                label="hybrid (BM25)"
              />
            </div>
          </div>

          {/* Reranking */}
          <SectionHeader title="Reranking" />
          <div className="grid grid-cols-3 gap-3">
            <div className="flex items-end pb-1">
              <Toggle
                value={Boolean(rer.enabled)}
                onChange={(v) => set(["reranking", "enabled"], v)}
                label="enabled"
              />
            </div>
            <Field label="model">
              <Input
                value={String(rer.model)}
                onChange={(v) => set(["reranking", "model"], v)}
              />
            </Field>
            <Field label="top_k">
              <Input
                type="number"
                value={Number(rer.top_k)}
                onChange={(v) => set(["reranking", "top_k"], Number(v))}
              />
            </Field>
            <Field label="candidate_cap">
              <Input
                type="number"
                value={Number(rer.candidate_cap)}
                onChange={(v) => set(["reranking", "candidate_cap"], Number(v))}
              />
            </Field>
          </div>

          {/* Generation */}
          <SectionHeader title="Generation" />
          <div className="grid grid-cols-3 gap-3">
            <Field label="response_type">
              <Select
                value={String(gen.response_type)}
                onChange={(v) => set(["generation", "response_type"], v)}
                options={["markdown", "json"]}
              />
            </Field>
            <Field label="temperature">
              <Input
                type="number"
                value={Number(gen.temperature)}
                onChange={(v) => set(["generation", "temperature"], Number(v))}
              />
            </Field>
            <Field label="max_retries">
              <Input
                type="number"
                value={Number(gen.max_retries)}
                onChange={(v) => set(["generation", "max_retries"], Number(v))}
              />
            </Field>
            <div className="flex items-end pb-1">
              <Toggle
                value={Boolean(gen.strict)}
                onChange={(v) => set(["generation", "strict"], v)}
                label="strict mode"
              />
            </div>
          </div>

          {/* Defaults */}
          <SectionHeader title="Defaults" />
          <Field label="system_prompt">
            <textarea
              value={String(def.system_prompt ?? "")}
              onChange={(e) => set(["defaults", "system_prompt"], e.target.value)}
              rows={4}
              className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none resize-y"
            />
          </Field>
        </div>
      )}

      <div className="flex gap-3 pt-1">
        <button
          onClick={handleSave}
          disabled={saving}
          className="rounded bg-indigo-700 px-4 py-2 text-xs text-white hover:bg-indigo-600 disabled:opacity-50 transition-colors"
        >
          {saving ? "Saving…" : isEdit ? "Update App" : "Register App"}
        </button>
        <button
          onClick={onSaved}
          className="rounded border border-gray-700 px-4 py-2 text-xs text-gray-400 hover:text-gray-200 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
