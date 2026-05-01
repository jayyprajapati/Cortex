import { useEffect, useState } from "react";
import { AppConfig, deleteApp, getApp, listApps } from "../api/client";

interface Props {
  onEdit: (appName: string | null) => void;
}

function PipelineTag({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded bg-gray-800 px-2 py-0.5 text-xs">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-200">{value}</span>
    </span>
  );
}

function AppCard({
  config,
  onEdit,
  onDelete,
}: {
  config: AppConfig;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const ing = config.ingestion as Record<string, unknown>;
  const emb = config.embedding as Record<string, unknown>;
  const ret = config.retrieval as Record<string, unknown>;
  const rer = config.reranking as Record<string, unknown>;
  const gen = config.generation as Record<string, unknown>;
  const taskNames = Object.keys(config.tasks ?? {});

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 flex flex-col gap-3">
      {/* Title row */}
      <div className="flex items-center justify-between">
        <div>
          <span className="text-indigo-300 font-semibold">{config.app_name}</span>
          <span className="ml-3 text-xs text-gray-500">
            collection: <span className="text-gray-300">{config.collection}</span>
          </span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onEdit}
            className="rounded border border-indigo-700 px-3 py-1 text-xs text-indigo-300 hover:bg-indigo-900 transition-colors"
          >
            Edit
          </button>
          <button
            onClick={onDelete}
            className="rounded border border-red-900 px-3 py-1 text-xs text-red-400 hover:bg-red-950 transition-colors"
          >
            Delete
          </button>
        </div>
      </div>

      {/* Pipeline badges */}
      <div className="flex flex-wrap gap-2">
        <PipelineTag label="strategy" value={String(ing.strategy ?? "—")} />
        <PipelineTag label="embed" value={String(emb.model ?? "—")} />
        <PipelineTag
          label="retrieval"
          value={`top_k=${ret.top_k} hybrid=${ret.hybrid} α=${ret.alpha}`}
        />
        <PipelineTag
          label="rerank"
          value={
            rer.enabled
              ? `${rer.model} top_k=${rer.top_k}`
              : "disabled"
          }
        />
        <PipelineTag
          label="gen"
          value={`${gen.response_type} t=${gen.temperature} strict=${gen.strict}`}
        />
      </div>

      {/* Tasks */}
      {taskNames.length > 0 && (
        <div className="text-xs text-gray-500">
          Tasks:{" "}
          {taskNames.map((t) => (
            <span
              key={t}
              className="mr-1 rounded bg-gray-800 px-2 py-0.5 text-gray-300"
            >
              {t}
              {t === config.default_task && (
                <span className="ml-1 text-indigo-400">★</span>
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function AppRegistryDashboard({ onEdit }: Props) {
  const [apps, setApps] = useState<AppConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const names = await listApps();
      const configs = await Promise.all(names.map((n) => getApp(n)));
      setApps(configs);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleDelete(appName: string) {
    if (!confirm(`Delete application "${appName}"?`)) return;
    try {
      await deleteApp(appName);
      await load();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-200">Registered Applications</h2>
        <div className="flex gap-2">
          <button
            onClick={load}
            className="rounded border border-gray-700 px-3 py-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={() => onEdit(null)}
            className="rounded bg-indigo-700 px-3 py-1 text-xs text-white hover:bg-indigo-600 transition-colors"
          >
            + New App
          </button>
        </div>
      </div>

      {loading && <p className="text-xs text-gray-500">Loading…</p>}
      {error && (
        <div className="rounded border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-300">
          {error}
        </div>
      )}

      {!loading && apps.length === 0 && (
        <p className="text-xs text-gray-500">No applications registered.</p>
      )}

      <div className="flex flex-col gap-3">
        {apps.map((cfg) => (
          <AppCard
            key={cfg.app_name}
            config={cfg}
            onEdit={() => onEdit(cfg.app_name)}
            onDelete={() => handleDelete(cfg.app_name)}
          />
        ))}
      </div>
    </div>
  );
}
