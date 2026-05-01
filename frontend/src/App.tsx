import { useState } from "react";
import AppRegistryDashboard from "./components/AppRegistryDashboard";
import AppConfigEditor from "./components/AppConfigEditor";
import CollectionManager from "./components/CollectionManager";
import DebugConsole from "./components/DebugConsole";

type Tab = "registry" | "editor" | "collections" | "debug";

const TABS: { id: Tab; label: string }[] = [
  { id: "registry", label: "App Registry" },
  { id: "editor", label: "Config Editor" },
  { id: "collections", label: "Collections" },
  { id: "debug", label: "Debug Console" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("registry");
  const [editTarget, setEditTarget] = useState<string | null>(null);

  function openEditor(appName: string | null) {
    setEditTarget(appName);
    setTab("editor");
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-3">
        <span className="text-indigo-400 font-bold text-base tracking-wide">
          CORTEX
        </span>
        <span className="text-gray-500 text-xs">admin v2</span>
      </header>

      {/* Tab bar */}
      <nav className="border-b border-gray-800 px-6 flex gap-1">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-xs border-b-2 transition-colors ${
              tab === t.id
                ? "border-indigo-400 text-indigo-300"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="flex-1 p-6 overflow-auto">
        {tab === "registry" && (
          <AppRegistryDashboard onEdit={openEditor} />
        )}
        {tab === "editor" && (
          <AppConfigEditor
            initialAppName={editTarget}
            onSaved={() => {
              setEditTarget(null);
              setTab("registry");
            }}
          />
        )}
        {tab === "collections" && <CollectionManager />}
        {tab === "debug" && <DebugConsole />}
      </main>
    </div>
  );
}
