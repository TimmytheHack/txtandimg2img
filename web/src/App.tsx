import React, { useEffect, useRef, useState } from 'react'

const WS_URL = import.meta.env.VITE_API_WS || 'ws://localhost:8000/ws/generate'

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [negative, setNegative] = useState('')
  const [width, setWidth] = useState(512)
  const [height, setHeight] = useState(768)
  const [steps, setSteps] = useState(24)
  const [guidance, setGuidance] = useState(7.0)
  const [seed, setSeed] = useState<string>('123456')

  const [imgSrc, setImgSrc] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<{ step: number, total: number } | null>(null)
  const [initImage, setInitImage] = useState<string | null>(null) // base64 PNG
  const [strength, setStrength] = useState(0.55)                  // 0.2–0.9 typical

  const onPickImage = (file: File | null) => {
    if (!file) { setInitImage(null); return }
    const reader = new FileReader()
    reader.onload = () => setInitImage(reader.result as string)
    reader.readAsDataURL(file)
  }
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'ready') {
          setBusy(false); setProgress(null)
        } else if (msg.type === 'started') {
          setBusy(true); setProgress({ step: 0, total: msg.total ?? steps })
        } else if (msg.type === 'progress') {
          if (typeof msg.step === 'number' && typeof msg.total === 'number') {
            setProgress({ step: msg.step, total: msg.total })
          }
        } else if (msg.type === 'final' && msg.image) {
          setImgSrc(`data:image/png;base64,${msg.image}`)
          setBusy(false); setProgress(null)
        } else if (msg.type === 'cancelled') {
          setBusy(false); setProgress(null)
        }
      } catch { /* ignore parse errors */ }
    }

    ws.onopen = () => { setBusy(false); setProgress(null) }
    ws.onclose = () => { wsRef.current = null; setBusy(false) }

    return () => { ws.close() }
  }, [])

  const sendOnce = () => {
    const p = prompt.trim()
    if (!p) return
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    setBusy(true)
    setProgress({ step: 0, total: steps })
    ws.send(JSON.stringify({
      prompt: p,
      steps, guidance, width, height, negative,
      seed: seed.trim(),
      image: initImage || undefined,   // send only if present
      strength
    }))
  }

  const stopNow = () => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ type: 'cancel' }))
    setBusy(false)
    setProgress(null)
  }

  const onKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendOnce()
    }
  }

  const lockSeed = () => setSeed((s) => (s && s.trim() !== '' ? s : `${Math.floor(Math.random()*1e9)}`))
  const randomizeSeed = () => setSeed(`${Math.floor(Math.random()*1e9)}`)

  const pct = progress ? Math.round((progress.step / Math.max(1, progress.total)) * 100) : 0

  return (
    <div className="min-h-dvh p-6 grid gap-6 lg:grid-cols-[420px_1fr]">
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Anime2D — Press to Generate</h1>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Describe your character… (Enter to generate, Shift+Enter for newline)"
          className="w-full h-40 p-3 rounded-xl bg-zinc-900 border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        
        <div className="col-span-2">
          <label className="text-sm text-zinc-400">Negative Prompt</label>
          <textarea
            value={negative}
            onChange={(e) => setNegative(e.target.value)}
            placeholder="What you DON'T want (e.g., extra fingers, low quality, blurry...)"
            className="mt-1 w-full h-20 p-2 rounded-xl bg-zinc-900 border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <label className="text-sm text-zinc-400">Reference image (PNG/JPG)</label>
            <div className="mt-1 flex items-center gap-3">
              <input type="file" accept="image/png,image/jpeg"
                onChange={(e) => onPickImage(e.target.files?.[0] ?? null)}
                className="block w-full text-sm text-zinc-300 file:mr-3 file:py-2 file:px-3 file:rounded-xl file:border-0 file:bg-zinc-800 file:text-zinc-200 hover:file:bg-zinc-700"
              />
              {initImage && (
                <button onClick={() => setInitImage(null)} className="px-3 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700">
                  Clear
                </button>
              )}
            </div>
            {initImage && (
              <div className="mt-2 flex items-center gap-3">
                <img src={initImage} className="h-20 rounded-md border border-zinc-800" />
                <div className="flex-1">
                  <label className="text-sm text-zinc-400">Image strength: {strength.toFixed(2)}</label>
                  <input type="range" min={0.1} max={0.95} step={0.05}
                    value={strength}
                    onChange={(e) => setStrength(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-xs text-zinc-500">
                    Lower = stick closer to the reference; higher = allow more change.
                  </div>
                </div>
              </div>
            )}
          </div>


          <div>
            <label className="text-sm text-zinc-400">Width</label>
            <input type="number" min={256} step={64}
              value={width} onChange={(e) => setWidth(parseInt(e.target.value || '0'))}
              className="mt-1 w-full p-2 rounded-xl bg-zinc-900 border border-zinc-700" />
          </div>

          <div>
            <label className="text-sm text-zinc-400">Height</label>
            <input type="number" min={256} step={64}
              value={height} onChange={(e) => setHeight(parseInt(e.target.value || '0'))}
              className="mt-1 w-full p-2 rounded-xl bg-zinc-900 border border-zinc-700" />
          </div>

          <div>
            <label className="text-sm text-zinc-400">Steps: {steps}</label>
            <input type="range" min={8} max={40} step={1}
              value={steps} onChange={(e) => setSteps(parseInt(e.target.value))}
              className="mt-2 w-full" />
          </div>

          <div>
            <label className="text-sm text-zinc-400">Guidance: {guidance.toFixed(1)}</label>
            <input type="range" min={4} max={12} step={0.1}
              value={guidance} onChange={(e) => setGuidance(parseFloat(e.target.value))}
              className="mt-2 w-full" />
          </div>

          <div className="col-span-2 flex items-center gap-2">
            <label className="text-sm text-zinc-400">Seed</label>
            <input
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="(blank = random each time)"
              className="w-48 p-2 rounded-xl bg-zinc-900 border border-zinc-700"
            />
            <button onClick={lockSeed} className="px-3 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700">Lock</button>
            <button onClick={randomizeSeed} className="px-3 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700">Random</button>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={sendOnce}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50"
            disabled={!prompt.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || busy}
          >
            {busy ? 'Generating…' : 'Generate (Enter)'}
          </button>

          <button
            onClick={stopNow}
            className="px-4 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50"
            disabled={!busy}
          >
            Stop
          </button>
        </div>

        {/* Progress bar */}
        <div className="h-2 w-full bg-zinc-800 rounded-full overflow-hidden">
          <div
            className="h-2 bg-indigo-500 transition-all"
            style={{ width: `${pct}%` }}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
          />
        </div>
        {progress && (
          <div className="text-xs text-zinc-400">
            Step {progress.step} / {progress.total} ({pct}%)
          </div>
        )}
      </div>

      <div className="flex items-start justify-center bg-zinc-900 border border-zinc-800 rounded-2xl p-4">
        {imgSrc ? (
          <img src={imgSrc} className="max-h-[80dvh] rounded-xl" />
        ) : (
          <div className="text-zinc-500">Enter a prompt and press generate…</div>
        )}
      </div>
    </div>
  )
}
