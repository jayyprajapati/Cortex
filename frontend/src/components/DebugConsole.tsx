import { useEffect, useState } from "react";
import { QueryResponse, listApps, query } from "../api/client";

interface Source {
  section: string;
  page: number | null;
  text?: string;
  score?: number;
  rerank_score?: number;
  hierarchy?: string;
}

function ScoreBar({ value }: { value: number | undefined }) {
  if (value === undefined) return null;
  const pct = Math.min(100, Math.round(value * 100));
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-24 bg-gray-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-400">{value.toFixed(3)}</span>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rounded border border-gray-800 bg-gray-900">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 text-left"
      >
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">#{index + 1}</span>
          <span className="text-xs text-gray-300">{source.section}</span>
          {source.page != null && (
            <span className="text-xs text-gray-500">p.{source.page}</span>
          )}
          {source.hierarchy && (
            <span className="text-xs text-gray-600 italic truncate max-w-xs">
              {source.hierarchy}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {source.rerank_score !== undefined && (
            <div className="flex items-center gap-1">
              <span className="text-xs text-gray-600">rerank</span>
              <ScoreBar value={source.rerank_score} />
            </div>
          )}
          {source.score !== undefined && (
            <div className="flex items-center gap-1">
              <span className="text-xs text-gray-600">hybrid</span>
              <ScoreBar value={source.score} />
            </div>
          )}
          <span className="text-xs text-gray-600">{expanded ? "▲" : "▼"}</span>
        </div>
      </button>

      {expanded && source.text && (
        <div className="border-t border-gray-800 px-3 py-2">
          <pre className="whitespace-pre-wrap text-xs text-gray-400 leading-relaxed">
            {source.text}
          </pre>
        </div>
      )}
    </div>
  );
}

export default function DebugConsole() {
  const [apps, setApps] = useState<string[]>([]);
  const [selectedApp, setSelectedApp] = useState("");
  const [userId, setUserId] = useState("debug_user");
  const [taskInput, setTaskInput] = useState("");
  const [docIds, setDocIds] = useState("");
  const [queryText, setQueryText] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  useEffect(() => {
    listApps().then((names) => {
      setApps(names);
      if (names.length > 0) setSelectedApp(names[0]);
    });
  }, []);

  async function handleQuery() {
    if (!selectedApp || !queryText.trim()) {
      setError("app_name and query are required.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    const ids = docIds
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const t0 = performance.now();
    try {
      const res = await query(
        selectedApp,
        userId,
        queryText,
        taskInput.trim() || undefined,
        ids.length > 0 ? ids : undefined
      );
      setResult(res);
      setElapsed(performance.now() - t0);
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "response" in e
          ? JSON.stringify((e as { response: { data: unknown } }).response.data, null, 2)
          : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  const isJson =
    result?.answer !== null &&
    typeof result?.answer === "object";

  return (
    <div className="flex flex-col gap-4 max-w-4xl">
      <h2 className="text-sm font-semibold text-gray-200">Debug Console</h2>

      {/* Controls */}
      <div className="rounded border border-gray-800 bg-gray-900 p-4 flex flex-col gap-3">
        <div className="grid grid-cols-4 gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Application</label>
            <select
              value={selectedApp}
              onChange={(e) => setSelectedApp(e.target.value)}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none"
            >
              {apps.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">user_id</label>
            <input
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">task (optional)</label>
            <input
              value={taskInput}
              onChange={(e) => setTaskInput(e.target.value)}
              placeholder="e.g. resume_match"
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">doc_ids (comma-sep)</label>
            <input
              value={docIds}
              onChange={(e) => setDocIds(e.target.value)}
              placeholder="id1, id2, …"
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-gray-500">Query</label>
          <textarea
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleQuery();
            }}
            rows={3}
            placeholder="Type your query… (⌘Enter to run)"
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none resize-none"
          />
        </div>

        <button
          onClick={handleQuery}
          disabled={loading}
          className="self-start rounded bg-indigo-700 px-4 py-2 text-xs text-white hover:bg-indigo-600 disabled:opacity-50 transition-colors"
        >
          {loading ? "Running…" : "Run Query"}
        </button>
      </div>

      {error && (
        <div className="rounded border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-300 whitespace-pre-wrap">
          {error}
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-4">
          {/* Timing */}
          {elapsed !== null && (
            <p className="text-xs text-gray-500">
              Response in {(elapsed / 1000).toFixed(2)}s
            </p>
          )}

          {/* Answer */}
          <div className="flex flex-col gap-1">
            <p className="text-xs font-semibold text-gray-400">Answer</p>
            <div className="rounded border border-gray-800 bg-gray-900 px-3 py-3">
              {isJson ? (
                <pre className="text-xs text-green-300 whitespace-pre-wrap">
                  {JSON.stringify(result.answer, null, 2)}
                </pre>
              ) : (
                <pre className="text-xs text-gray-200 whitespace-pre-wrap leading-relaxed">
                  {String(result.answer)}
                </pre>
              )}
            </div>
          </div>

          {/* Sources */}
          {result.sources.length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="text-xs font-semibold text-gray-400">
                Sources ({result.sources.length})
              </p>
              {result.sources.map((src, i) => (
                <SourceCard key={i} source={src} index={i} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
