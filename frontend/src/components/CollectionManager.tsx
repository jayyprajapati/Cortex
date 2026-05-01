import { useEffect, useState } from "react";
import {
  CollectionInfo,
  createCollection,
  deleteCollection,
  listCollections,
} from "../api/client";

export default function CollectionManager() {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [newSize, setNewSize] = useState("384");
  const [creating, setCreating] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setCollections(await listCollections());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleCreate() {
    const name = newName.trim().toLowerCase();
    const size = parseInt(newSize, 10);
    if (!name || isNaN(size) || size < 1) {
      setError("Provide a valid collection name and vector size.");
      return;
    }
    setCreating(true);
    setError(null);
    try {
      await createCollection(name, size);
      setNewName("");
      setNewSize("384");
      await load();
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "response" in e
          ? JSON.stringify((e as { response: { data: unknown } }).response.data)
          : String(e);
      setError(msg);
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(name: string) {
    if (!confirm(`Delete collection "${name}"? This is irreversible.`)) return;
    try {
      await deleteCollection(name);
      await load();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="flex flex-col gap-4 max-w-2xl">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-200">Qdrant Collections</h2>
        <button
          onClick={load}
          className="rounded border border-gray-700 px-3 py-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="rounded border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-300 whitespace-pre-wrap">
          {error}
        </div>
      )}

      {/* Create form */}
      <div className="rounded border border-gray-800 bg-gray-900 p-4 flex flex-col gap-3">
        <p className="text-xs font-semibold text-gray-400">Create Collection</p>
        <div className="flex gap-2 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Name</label>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="e.g. myapp"
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none w-36"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-500">Vector Size</label>
            <input
              type="number"
              value={newSize}
              onChange={(e) => setNewSize(e.target.value)}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-gray-100 focus:border-indigo-500 focus:outline-none w-24"
            />
          </div>
          <button
            onClick={handleCreate}
            disabled={creating}
            className="rounded bg-indigo-700 px-3 py-1 text-xs text-white hover:bg-indigo-600 disabled:opacity-50 transition-colors"
          >
            {creating ? "Creating…" : "Create"}
          </button>
        </div>
      </div>

      {/* Collections list */}
      {loading && <p className="text-xs text-gray-500">Loading…</p>}
      {!loading && collections.length === 0 && (
        <p className="text-xs text-gray-500">No collections found.</p>
      )}

      <div className="flex flex-col gap-2">
        {collections.map((col) => (
          <div
            key={col.name}
            className="flex items-center justify-between rounded border border-gray-800 bg-gray-900 px-4 py-3"
          >
            <div className="flex items-center gap-4">
              <span className="text-indigo-300 text-xs font-semibold">{col.name}</span>
              <span className="text-xs text-gray-500">
                {col.points_count?.toLocaleString() ?? "—"} vectors
              </span>
              <span className="text-xs text-gray-500">
                dim: {col.vector_size ?? "—"}
              </span>
              <span
                className={`text-xs rounded px-1.5 py-0.5 ${
                  col.status === "green" || col.status === "CollectionStatus.Green"
                    ? "bg-green-950 text-green-400"
                    : "bg-gray-800 text-gray-400"
                }`}
              >
                {col.status}
              </span>
            </div>
            <button
              onClick={() => handleDelete(col.name)}
              className="rounded border border-red-900 px-2 py-1 text-xs text-red-400 hover:bg-red-950 transition-colors"
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
