import { useEffect, useRef, useState } from "react";
import { Niivue, SLICE_TYPE, SHOW_RENDER } from "@niivue/niivue";
import { studyFetch } from "../study-access";

type Viewing = { volumes: string[]; outlineCenterMm: [number, number, number] | null };
export default function MriViewer({ id, viewing }: { id: string; viewing: Viewing }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const comparison = useRef<HTMLCanvasElement>(null);
  const peer = useRef<Niivue | null>(null);
  const savedContrast = useRef<{ modality: string; min: number; max: number } | null>(null);
  const viewer = useRef<Niivue | null>(null);
  const [modality, setModality] = useState("flair");
  const [mask, setMask] = useState("model");
  const [opacity, setOpacity] = useState(0.45);
  const [plane, setPlane] = useState(() => window.innerWidth < 700 ? SLICE_TYPE.AXIAL : SLICE_TYPE.MULTIPLANAR);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const syncPeer = () => { const nv = viewer.current; const other = peer.current; if (nv && other) { nv.doSync2d(other); other.drawScene(); } };
  useEffect(() => { peer.current?.setOpacity(1, opacity); peer.current?.setSliceType(plane); }, [opacity, plane]);
  useEffect(() => {
    let disposed = false;
    const controller = new AbortController();
    const urls: string[] = [];
    const nv = new Niivue({ dragAndDropEnabled: false, isNearestInterpolation: true, multiplanarShowRender: SHOW_RENDER.NEVER });
    let other: Niivue | null = null;
    const previous = viewer.current?.scene.crosshairPos;
    viewer.current = nv;
    setLoading(true); setError("");
    const load = async () => {
      await nv.attachToCanvas(canvas.current!);
      if (disposed) return;
      const paths = [`/api/studies/${id}/viewing/${modality}`, mask !== "reference" ? `/api/studies/${id}/artifacts/segmentation` : `/api/studies/${id}/viewing/reference`];
      for (const path of paths) {
        const response = await studyFetch(path, { signal: controller.signal });
        if (!response.ok) throw new Error("Viewing files unavailable. Check the study expiry.");
        const blob = await response.blob();
        if (disposed) return;
        urls.push(URL.createObjectURL(blob));
      }
      if (disposed) return;
      await nv.loadVolumes(urls.map((url, index) => ({ url, name: `${index}.nii.gz`, colormap: index ? "red" : "gray", opacity: index ? opacity : 1 })));
      if (disposed) return;
      if (savedContrast.current?.modality === modality) {
        nv.volumes[0].cal_min = savedContrast.current.min; nv.volumes[0].cal_max = savedContrast.current.max;
        nv.updateGLVolume();
      }
      nv.setSliceType(plane);
      if (previous) nv.scene.crosshairPos = previous;
      if (mask === "compare" && comparison.current) {
        other = new Niivue({ dragAndDropEnabled: false, isNearestInterpolation: true, multiplanarShowRender: SHOW_RENDER.NEVER });
        peer.current = other;
        await other.attachToCanvas(comparison.current);
        const response = await studyFetch(`/api/studies/${id}/viewing/reference`, { signal: controller.signal });
        if (!response.ok) throw new Error("Reference unavailable");
        const blob = await response.blob(); if (disposed) return;
        const referenceUrl = URL.createObjectURL(blob); urls.push(referenceUrl);
        await other.loadVolumes([{ url: urls[0], name: "scan.nii.gz" }, { url: referenceUrl, name: "reference.nii.gz", colormap: "blue", opacity }]);
        other.setSliceType(plane);
        other.volumes[0].cal_min = nv.volumes[0].cal_min; other.volumes[0].cal_max = nv.volumes[0].cal_max; other.updateGLVolume();
        nv.broadcastTo(other, { '2d': true, '3d': false }); other.broadcastTo(nv, { '2d': true, '3d': false });
        nv.syncOpts.cal_min = true; nv.syncOpts.cal_max = true;
        other.syncOpts.cal_min = true; other.syncOpts.cal_max = true;
        nv.doSync2d(other); other.drawScene();
      }
      nv.drawScene();
      setLoading(false);
    };
    void load().catch(reason => { if (!disposed) { setError(`Viewer could not load: ${reason.message}. Downloads remain available.`); setLoading(false); } });
    return () => { disposed = true; controller.abort(); if (typeof nv.volumes[0]?.cal_min === "number" && typeof nv.volumes[0]?.cal_max === "number") savedContrast.current = { modality, min: nv.volumes[0].cal_min, max: nv.volumes[0].cal_max }; urls.forEach(url => URL.revokeObjectURL(url)); nv.cleanup(); other?.cleanup(); peer.current = null; };
  }, [id, modality, mask]);
  return <section className="mri-viewer" aria-label="MRI outline viewer">
    <div className="viewer-controls">
      <label>Scan <select value={modality} onChange={event => setModality(event.target.value)}>{["t1", "t1ce", "t2", "flair"].map(name => <option key={name} value={name}>{name.toUpperCase()}</option>)}</select></label>
      <label>View <select value={plane} onChange={event => { const value = Number(event.target.value) as SLICE_TYPE; setPlane(value); viewer.current?.setSliceType(value); }}>{[[SLICE_TYPE.MULTIPLANAR, "Three planes"], [SLICE_TYPE.AXIAL, "Axial"], [SLICE_TYPE.CORONAL, "Coronal"], [SLICE_TYPE.SAGITTAL, "Sagittal"]].map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      {viewing.volumes.includes("reference") && <label>Outline <select value={mask} onChange={event => setMask(event.target.value)}><option value="model">Model</option><option value="reference">Expert reference</option><option value="compare">Compare side by side</option></select></label>}
      <label>Opacity <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={event => { const value = Number(event.target.value); setOpacity(value); viewer.current?.setOpacity(1, value); }} /></label>
      <button disabled={!viewing.outlineCenterMm || loading} onClick={() => { const nv = viewer.current; if (nv && viewing.outlineCenterMm) { nv.scene.crosshairPos = nv.mm2frac(viewing.outlineCenterMm); nv.drawScene(); syncPeer(); } }}>Jump to outline</button>
      <button onClick={() => { const nv = viewer.current; if (nv) { nv.scene.crosshairPos = [0.5, 0.5, 0.5]; nv.drawScene(); syncPeer(); } }}>Reset position</button>
      {(["X", "Y", "Z"] as const).map((axis, index) => <span key={axis}>{axis} slice <button aria-label={`Previous ${axis} slice`} onClick={() => { const step = [0, 0, 0]; step[index] = -1; viewer.current?.moveCrosshairInVox(step[0], step[1], step[2]); syncPeer(); }}>−</button><button aria-label={`Next ${axis} slice`} onClick={() => { const step = [0, 0, 0]; step[index] = 1; viewer.current?.moveCrosshairInVox(step[0], step[1], step[2]); syncPeer(); }}>+</button></span>)}
    </div>
    <p>Scroll over a plane to move through slices. Drag to move the crosshair; right-drag to adjust brightness and contrast.</p>
    {loading && <p role="status">Loading scan…</p>}{error && <p role="alert">{error}</p>}
    <div className={mask === "compare" ? "viewer-pair" : ""}><div><strong>{mask === "reference" ? "Expert reference" : "Model outline"}</strong><div className="viewer-canvas"><canvas ref={canvas} aria-label="MRI slices with selected outline overlay" /></div></div>{mask === "compare" && <div><strong>Expert reference</strong><div className="viewer-canvas"><canvas ref={comparison} aria-label="Expert reference comparison" /></div></div>}</div>
  </section>;
}
