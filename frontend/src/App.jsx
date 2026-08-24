import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import {
  Brain, Activity, Shield, FileText, Upload, RefreshCw,
  CheckCircle, AlertTriangle, X, Info, Cpu, Database,
  ArrowUpRight, Sparkles, User, Calendar, Layers, ZoomIn,
  Loader2, FlaskConical, Stethoscope, ChevronRight, HelpCircle
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const DEMO_CASES = [
  {
    id: 1,
    name: 'Patient A — High Grade Glial Lesion',
    diagnosis_hint: 'Glioma Suspected',
    patient: 'Eleanor Vance, 54F',
    notes: 'Frontal lobe axial T1 contrast-enhanced MRI scan.',
    fileUrl: '/samples/glioma.jpg',
    filename: 'glioma_sample.jpg'
  },
  {
    id: 2,
    name: 'Patient B — Dural Convexity Mass',
    diagnosis_hint: 'Meningioma Suspected',
    patient: 'Arthur Pendelton, 62M',
    notes: 'Parasagittal extra-axial mass with dural tail.',
    fileUrl: '/samples/meningioma.jpg',
    filename: 'meningioma_sample.jpg'
  },
  {
    id: 3,
    name: 'Patient C — Sellar / Suprasellar Mass',
    diagnosis_hint: 'Pituitary Adenoma Suspected',
    patient: 'Clara Oswald, 39F',
    notes: 'Coronal T2 weighted sellar region enlargement.',
    fileUrl: '/samples/pituitary.jpg',
    filename: 'pituitary_sample.jpg'
  },
  {
    id: 4,
    name: 'Patient D — Unremarkable Baseline Scan',
    diagnosis_hint: 'Healthy Brain Tissue',
    patient: 'James Wilson, 28M',
    notes: 'Symmetrical cerebral hemispheres, no mass effect.',
    fileUrl: '/samples/healthy.jpg',
    filename: 'healthy_sample.jpg'
  }
];

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [patientName, setPatientName] = useState('John Doe');
  const [patientAge, setPatientAge] = useState(48);
  const [scanType, setScanType] = useState('Axial T1 Contrast');
  const [clinicalNotes, setClinicalNotes] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedDemoId, setSelectedDemoId] = useState(null);

  // Modals
  const [showStatsModal, setShowStatsModal] = useState(false);
  const [showSafetyModal, setShowSafetyModal] = useState(false);
  const [backendStatus, setBackendStatus] = useState({ online: false, device: 'CPU' });

  const fileInputRef = useRef(null);

  useEffect(() => {
    // Check backend health
    axios.get(`${API_BASE}/health`)
      .then(res => {
        setBackendStatus({ online: res.data.status === 'ok', device: res.data.device || 'CPU' });
      })
      .catch(() => {
        setBackendStatus({ online: false, device: 'Offline' });
      });
  }, []);

  const resetAll = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setSelectedDemoId(null);
    setPatientName('John Doe');
    setClinicalNotes('');
  };

  const handleFileSelect = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
      setSelectedDemoId(null);
    }
  };

  const loadDemoCase = async (demo) => {
    setSelectedDemoId(demo.id);
    setPatientName(demo.patient.split(',')[0]);
    setClinicalNotes(demo.notes);
    setResult(null);
    setError(null);

    try {
      const response = await fetch(demo.fileUrl);
      const blob = await response.blob();
      const demoFile = new File([blob], demo.filename, { type: 'image/jpeg' });
      setFile(demoFile);
      setPreview(demo.fileUrl);
    } catch (err) {
      console.error('Failed to load demo sample', err);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE}/predict`, formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Inference failed. Ensure backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const downloadPDF = () => {
    if (!result) return;
    try {
      const doc = new jsPDF();
      
      // Header Banner (Teal)
      doc.setFillColor(13, 127, 130);
      doc.rect(0, 0, 210, 26, 'F');
      
      doc.setFontSize(16);
      doc.setTextColor(255, 255, 255);
      doc.text("NEUROSCAN AI — CLINICAL ONCOLOGY REPORT", 14, 17);

      // Metadata
      doc.setFontSize(10);
      doc.setTextColor(71, 85, 105);
      doc.text(`Patient Name: ${patientName || "Anonymous"}`, 14, 38);
      doc.text(`Age: ${patientAge} yrs   |   Scan Type: ${scanType}`, 14, 44);
      doc.text(`Report Timestamp: ${new Date().toLocaleString()}`, 14, 50);

      // Results Table
      autoTable(doc, {
        startY: 58,
        head: [['Metric / Clinical Parameter', 'AI Diagnostic Finding']],
        body: [
          ['Primary Classification', result.diagnosis],
          ['Confidence Index', `${result.confidence}%`],
          ['Tissue Assessment', result.status],
          ['Model Architecture', 'EfficientNet-B0 Deep CNN'],
          ['Clinical Notes', clinicalNotes || 'Standard cranial MRI evaluation']
        ],
        headStyles: { fillColor: [13, 127, 130], textColor: [255, 255, 255], fontStyle: 'bold' },
        alternateRowStyles: { fillColor: [248, 251, 251] },
        styles: { fontSize: 10, cellPadding: 5 }
      });

      const finalY = doc.lastAutoTable?.finalY || 130;
      
      doc.setFontSize(9);
      doc.setTextColor(148, 163, 184);
      doc.text("Disclaimer: Automated computer-vision assistance tool. Final diagnosis requires board-certified radiologist verification.", 14, finalY + 20);

      doc.save(`NeuroScan_Report_${result.diagnosis}_${patientName.replace(/\s+/g, '_')}.pdf`);
    } catch (err) {
      console.error(err);
      alert("Failed to generate PDF. Please verify browser settings.");
    }
  };

  const getStatusBadge = () => {
    if (!result) return null;
    if (result.diagnosis === 'HEALTHY') {
      return {
        label: 'Healthy Brain Scan',
        color: 'bg-emerald-50 text-emerald-700 border-emerald-200 shadow-sm',
        icon: CheckCircle
      };
    }
    if (result.diagnosis === 'INCONCLUSIVE') {
      return {
        label: 'Inconclusive / Artifact Detected',
        color: 'bg-amber-50 text-amber-700 border-amber-200 shadow-sm',
        icon: AlertTriangle
      };
    }
    return {
      label: `Tumor Detected: ${result.diagnosis}`,
      color: 'bg-rose-50 text-rose-700 border-rose-200 shadow-sm',
      icon: AlertTriangle
    };
  };

  const badge = getStatusBadge();

  return (
    <div className="flex h-screen w-screen bg-gradient-to-br from-[#EAF5F5] via-[#e0f2f2] to-[#cce8e8] font-sans overflow-hidden print:h-auto print:w-auto print:bg-white">
      {/* Container Card — Fills viewport */}
      <div className="w-full h-full bg-white flex overflow-hidden print:block print:h-auto">

        {/* ── Sidebar 1: Dark Teal Icon Navigation ────────────────────────── */}
        <div className="w-[85px] bg-gradient-to-b from-[#0c7a7d] via-[#0D7F82] to-[#075558] flex flex-col items-center py-7 gap-7 flex-shrink-0 print:hidden select-none">
          <div onClick={resetAll} title="Home / Reset" className="flex flex-col items-center gap-1.5 cursor-pointer group relative">
            <div className="p-3.5 bg-white rounded-2xl text-[#0D7F82] shadow-lg transform group-hover:scale-110 transition-all duration-200">
              <Brain className="w-5 h-5" />
            </div>
            <span className="text-[9px] text-white font-semibold tracking-wide uppercase">Scan</span>
          </div>

          <div onClick={() => setShowStatsModal(true)} title="Model Specifications" className="flex flex-col items-center gap-1.5 cursor-pointer group">
            <div className="p-3.5 text-white/60 group-hover:text-white group-hover:bg-white/10 rounded-2xl transition-all">
              <Database className="w-5 h-5" />
            </div>
            <span className="text-[9px] text-white/60 group-hover:text-white font-medium tracking-wide uppercase transition-colors">Model</span>
          </div>

          <div onClick={() => setShowSafetyModal(true)} title="Safety & Methodology" className="flex flex-col items-center gap-1.5 cursor-pointer group">
            <div className="p-3.5 text-white/60 group-hover:text-white group-hover:bg-white/10 rounded-2xl transition-all">
              <Shield className="w-5 h-5" />
            </div>
            <span className="text-[9px] text-white/60 group-hover:text-white font-medium tracking-wide uppercase transition-colors">Safety</span>
          </div>

          {result && (
            <div onClick={downloadPDF} title="Export PDF Report" className="flex flex-col items-center gap-1.5 cursor-pointer group">
              <div className="p-3.5 text-white/60 group-hover:text-white group-hover:bg-white/10 rounded-2xl transition-all">
                <FileText className="w-5 h-5" />
              </div>
              <span className="text-[9px] text-white/60 group-hover:text-white font-medium tracking-wide uppercase transition-colors">Export</span>
            </div>
          )}
        </div>

        {/* ── Sidebar 2: Patient Context & Demo Cases ───────────────────────── */}
        <div className="w-[330px] bg-white border-r border-gray-100/80 flex flex-col overflow-y-auto custom-scrollbar flex-shrink-0 print:hidden">
          <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#0D7F82] to-[#129A9E] flex items-center justify-center shadow-sm">
                <FlaskConical className="w-4 h-4 text-white" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-gray-800 tracking-tight">Patient Context</h2>
                <p className="text-[10px] text-gray-400">Clinical details & presets</p>
              </div>
            </div>

            {/* Demo Cases Preset Buttons */}
            <div>
              <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.15em] mb-2.5">
                Benchmark Demo Cases
              </h3>
              <div className="space-y-2">
                {DEMO_CASES.map((dc) => (
                  <button
                    key={dc.id}
                    onClick={() => loadDemoCase(dc)}
                    className={`w-full text-left p-3 rounded-2xl border transition-all duration-200 group ${
                      selectedDemoId === dc.id
                        ? 'bg-gradient-to-r from-[#EAF5F5] to-[#f4fbfb] border-[#0D7F82]/40 shadow-[0_4px_15px_-3px_rgba(13,127,130,0.15)] translate-x-0.5'
                        : 'bg-gray-50/80 border-gray-100 hover:border-[#0D7F82]/30 hover:bg-[#EAF5F5]/40 hover:-translate-y-0.5'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <div className={`w-7 h-7 rounded-xl flex items-center justify-center text-xs font-bold transition-all shadow-sm ${
                        selectedDemoId === dc.id
                          ? 'bg-[#0D7F82] text-white'
                          : 'bg-[#d2ecec] text-[#0D7F82] group-hover:bg-[#0D7F82] group-hover:text-white'
                      }`}>
                        {dc.id}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="font-semibold text-gray-800 text-[11px] leading-tight truncate">
                          {dc.name}
                        </div>
                        <div className="text-[10px] text-gray-400 mt-0.5 truncate">
                          {dc.diagnosis_hint}
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Patient Context Inputs */}
            <div>
              <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-[0.15em] mb-2.5">
                Patient Parameters
              </h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-[11px] font-semibold text-gray-500 mb-1">
                    Patient Identifier / Name
                  </label>
                  <input
                    type="text"
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    placeholder="e.g. Eleanor Vance"
                    className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-xl text-xs text-gray-800 focus:border-[#0D7F82] focus:ring-2 focus:ring-[#0D7F82]/10 outline-none transition-all font-medium"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-[11px] font-semibold text-gray-500 mb-1">
                      Age (Years)
                    </label>
                    <input
                      type="number"
                      value={patientAge}
                      onChange={(e) => setPatientAge(Number(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-xl text-xs text-gray-800 focus:border-[#0D7F82] focus:ring-2 focus:ring-[#0D7F82]/10 outline-none transition-all font-medium"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-gray-500 mb-1">
                      Modality
                    </label>
                    <select
                      value={scanType}
                      onChange={(e) => setScanType(e.target.value)}
                      className="w-full px-2.5 py-2 bg-gray-50 border border-gray-200 rounded-xl text-xs text-gray-700 focus:border-[#0D7F82] focus:ring-2 focus:ring-[#0D7F82]/10 outline-none transition-all"
                    >
                      <option>Axial T1 Contrast</option>
                      <option>Coronal T2</option>
                      <option>Sagittal FLAIR</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-[11px] font-semibold text-gray-500 mb-1">
                    Radiology Notes / History
                  </label>
                  <textarea
                    rows={2}
                    value={clinicalNotes}
                    onChange={(e) => setClinicalNotes(e.target.value)}
                    placeholder="Relevant symptoms or clinical indications..."
                    className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-xl text-xs text-gray-700 focus:border-[#0D7F82] focus:ring-2 focus:ring-[#0D7F82]/10 outline-none transition-all resize-none"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Main Content Area ────────────────────────────────────────────── */}
        <div className="flex-1 bg-[#F8FBFB] flex flex-col relative min-w-0 overflow-y-auto custom-scrollbar print:bg-white print:p-0">

          {/* Header Bar */}
          <div className="h-16 px-7 flex items-center justify-between border-b border-gray-100 bg-white/70 backdrop-blur-sm shrink-0 print:hidden">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-gradient-to-br from-[#0D7F82] to-[#129A9E] rounded-xl flex items-center justify-center shadow-md">
                <Activity className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-sm font-bold text-gray-800 leading-tight">
                  NeuroScan AI Oncology Engine
                </h1>
                <p className="text-[10px] text-gray-400 font-medium tracking-wide">
                  Brain Tumor Detection & Categorization (EfficientNet-B0)
                </p>
              </div>
            </div>

            {/* Status Pills */}
            <div className="flex items-center gap-2">
              <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold tracking-wider border ${
                backendStatus.online
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : 'bg-rose-50 text-rose-700 border-rose-200'
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${backendStatus.online ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
                {backendStatus.online ? `Model Online (${backendStatus.device})` : 'Backend Offline'}
              </div>

              <button
                onClick={() => setShowStatsModal(true)}
                className="w-8 h-8 rounded-xl bg-gray-100 text-gray-500 flex items-center justify-center hover:bg-gray-200 hover:text-gray-700 transition-all"
              >
                <Info className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="mx-6 mt-4 p-3.5 bg-rose-50 border border-rose-200 rounded-2xl text-rose-700 text-xs flex items-center gap-2 animate-fade-in">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Core Diagnostic Grid */}
          <div className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 items-start">
            {/* ── Left Column: MRI Input & Scan Zone (5 cols) ───────────────── */}
            <div className="lg:col-span-5 bg-white border border-gray-100/80 rounded-2xl p-5 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-[#0D7F82]" />
                  <h3 className="text-xs font-bold uppercase tracking-wider text-gray-800">
                    MRI Scan Input
                  </h3>
                </div>
                {file && (
                  <button
                    onClick={resetAll}
                    className="text-[11px] text-gray-400 hover:text-[#0D7F82] font-medium flex items-center gap-1 transition-colors"
                  >
                    <RefreshCw className="w-3 h-3" /> Clear
                  </button>
                )}
              </div>

              {/* Upload Drop Area */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => handleFileSelect(e.target.files[0])}
                className="hidden"
              />

              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  handleFileSelect(e.dataTransfer.files[0]);
                }}
                className={`relative border-2 border-dashed rounded-2xl p-5 text-center min-h-[300px] flex flex-col items-center justify-center cursor-pointer transition-all duration-200 ${
                  isDragging
                    ? 'border-[#0D7F82] bg-[#EAF5F5]/60 scale-[1.01]'
                    : preview
                    ? 'border-gray-200 bg-gray-900/5'
                    : 'border-gray-200 bg-gray-50/60 hover:bg-[#EAF5F5]/20 hover:border-[#0D7F82]/40'
                }`}
              >
                {preview ? (
                  <div className="relative group max-h-[290px] w-full flex items-center justify-center">
                    <img
                      src={preview}
                      alt="MRI Scan Preview"
                      className="max-h-[280px] w-auto object-contain rounded-xl shadow-md border border-gray-200/80"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 rounded-xl flex items-center justify-center transition-opacity text-white text-xs font-medium gap-1.5">
                      <ZoomIn className="w-4 h-4" /> Click to change image
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3 py-8">
                    <div className="w-12 h-12 rounded-2xl bg-[#d2ecec] text-[#0D7F82] mx-auto flex items-center justify-center shadow-sm">
                      <Upload className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="text-xs font-bold text-gray-700">
                        Drop high-resolution Brain MRI scan
                      </p>
                      <p className="text-[11px] text-gray-400 mt-0.5">
                        Supports axial, coronal, and sagittal DICOM/JPEG/PNG
                      </p>
                    </div>
                    <span className="inline-block px-3 py-1 rounded-lg bg-white border border-gray-200 text-[10px] font-semibold text-gray-500 shadow-sm">
                      Browse Files
                    </span>
                  </div>
                )}
              </div>

              {/* Action Button */}
              <button
                onClick={handleAnalyze}
                disabled={!file || loading}
                className={`w-full py-3 px-4 rounded-xl font-bold text-xs tracking-wider uppercase flex items-center justify-center gap-2 transition-all shadow-md ${
                  loading
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : file
                    ? 'bg-gradient-to-r from-[#0c7a7d] to-[#0D7F82] text-white hover:shadow-[0_4px_15px_-3px_rgba(13,127,130,0.4)] hover:-translate-y-0.5 active:translate-y-0'
                    : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-[#0D7F82]" />
                    Analyzing Neural Slices...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Execute Deep Diagnostic Inference
                  </>
                )}
              </button>
            </div>

            {/* ── Right Column: Diagnostic Intelligence Results (7 cols) ──── */}
            <div className="lg:col-span-7 space-y-4">
              {result ? (
                <div className="space-y-4 animate-fade-in">
                  {/* Primary Diagnosis Card */}
                  <div className="bg-white border border-gray-100/80 rounded-2xl p-6 shadow-sm space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Stethoscope className="w-4 h-4 text-[#0D7F82]" />
                        <h3 className="text-xs font-bold uppercase tracking-wider text-gray-800">
                          Clinical Findings
                        </h3>
                      </div>

                      {badge && (
                        <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${badge.color}`}>
                          <badge.icon className="w-3.5 h-3.5" />
                          {badge.label}
                        </div>
                      )}
                    </div>

                    {/* Diagnostic Metric Highlights */}
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                      {/* Classification Category */}
                      <div className="p-4 rounded-xl bg-gray-50/80 border border-gray-100 space-y-1">
                        <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                          Tumor Class
                        </span>
                        <div className="text-lg font-bold text-gray-900 font-mono">
                          {result.diagnosis}
                        </div>
                        <div className="text-[10px] text-gray-400">
                          {result.diagnosis === 'GLIOMA' && 'Intra-axial glial tissue origin'}
                          {result.diagnosis === 'MENINGIOMA' && 'Extra-axial meningeal origin'}
                          {result.diagnosis === 'PITUITARY' && 'Sellar / adenoma origin'}
                          {result.diagnosis === 'HEALTHY' && 'No neoplastic lesion identified'}
                          {result.diagnosis === 'INCONCLUSIVE' && 'Confidence threshold not met'}
                        </div>
                      </div>

                      {/* Confidence Score */}
                      <div className="p-4 rounded-xl bg-gray-50/80 border border-gray-100 space-y-1">
                        <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                          Model Confidence
                        </span>
                        <div className="text-lg font-bold text-[#0D7F82] font-mono">
                          {result.confidence}%
                        </div>
                        {/* Progress Bar */}
                        <div className="w-full bg-gray-200 h-1.5 rounded-full overflow-hidden mt-1">
                          <div
                            className="bg-[#0D7F82] h-full rounded-full transition-all duration-700"
                            style={{ width: `${result.confidence}%` }}
                          />
                        </div>
                      </div>

                      {/* Diagnostic Status */}
                      <div className="p-4 rounded-xl bg-gray-50/80 border border-gray-100 space-y-1">
                        <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                          Tissue Status
                        </span>
                        <div className={`text-sm font-bold mt-1 ${
                          result.diagnosis === 'HEALTHY'
                            ? 'text-emerald-600'
                            : result.diagnosis === 'INCONCLUSIVE'
                            ? 'text-amber-600'
                            : 'text-rose-600'
                        }`}>
                          {result.status}
                        </div>
                        <div className="text-[10px] text-gray-400">
                          Safety Floor: &gt;75%
                        </div>
                      </div>
                    </div>

                    {/* Warning if Inconclusive */}
                    {result.warning && (
                      <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-start gap-2.5">
                        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                        <div>
                          <p className="font-bold">Low Diagnostic Confidence Warning</p>
                          <p className="mt-0.5 text-amber-700">{result.warning}</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Radiologist Narrative & Export Card */}
                  <div className="bg-white border border-gray-100/80 rounded-2xl p-5 shadow-sm space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold uppercase tracking-wider text-gray-700">
                        Patient Summary & Actions
                      </span>
                      <span className="text-[11px] text-gray-400 font-mono">
                        ID: {patientName} · {scanType}
                      </span>
                    </div>

                    <div className="p-3.5 rounded-xl bg-[#F8FBFB] border border-gray-100 text-xs text-gray-600 leading-relaxed space-y-1">
                      <p>
                        <strong>Clinical Impression:</strong> Automated analysis of the submitted MRI slice classified the lesion as{' '}
                        <span className="font-bold text-[#0D7F82]">{result.diagnosis}</span> with a certainty index of{' '}
                        <span className="font-bold text-[#0D7F82]">{result.confidence}%</span>.
                      </p>
                      {clinicalNotes && (
                        <p className="text-gray-500 pt-1">
                          <strong>Physician Notes:</strong> {clinicalNotes}
                        </p>
                      )}
                    </div>

                    {/* Export / Print Buttons */}
                    <div className="flex items-center gap-2.5 pt-1">
                      <button
                        onClick={downloadPDF}
                        className="flex-1 py-2.5 px-4 rounded-xl bg-gray-50 hover:bg-[#EAF5F5] border border-gray-200 hover:border-[#0D7F82]/40 text-[#0D7F82] text-xs font-bold flex items-center justify-center gap-1.5 transition-all shadow-sm"
                      >
                        <FileText className="w-4 h-4" />
                        Download Clinical PDF
                      </button>

                      <button
                        onClick={() => window.print()}
                        className="py-2.5 px-4 rounded-xl bg-gray-50 hover:bg-gray-100 border border-gray-200 text-gray-600 text-xs font-semibold flex items-center justify-center gap-1.5 transition-all shadow-sm"
                      >
                        Print Summary
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                /* Empty / Ready State Guidance Card */
                <div className="bg-white border border-gray-100/80 rounded-2xl p-8 shadow-sm space-y-5">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-2xl bg-[#EAF5F5] text-[#0D7F82] flex items-center justify-center">
                      <Activity className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-gray-800">
                        Diagnostic System Ready
                      </h3>
                      <p className="text-xs text-gray-400">
                        Select a benchmark demo case or upload a cranial MRI scan to start.
                      </p>
                    </div>
                  </div>

                  {/* 3-Step Guide */}
                  <div className="space-y-3 text-xs text-gray-600">
                    <div className="p-3.5 rounded-xl bg-gray-50/80 border border-gray-100 flex items-start gap-3">
                      <span className="w-5 h-5 rounded-full bg-[#d2ecec] text-[#0D7F82] text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                        1
                      </span>
                      <div>
                        <p className="font-bold text-gray-800">Cranial Slice Ingestion</p>
                        <p className="text-gray-500 mt-0.5">
                          Load an axial or coronal MRI scan from the left sidebar presets or drop your own image.
                        </p>
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-gray-50/80 border border-gray-100 flex items-start gap-3">
                      <span className="w-5 h-5 rounded-full bg-[#d2ecec] text-[#0D7F82] text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                        2
                      </span>
                      <div>
                        <p className="font-bold text-gray-800">Deep Feature Extraction</p>
                        <p className="text-gray-500 mt-0.5">
                          EfficientNet-B0 extracts multi-scale convolutional feature maps across 4 distinct oncology classes.
                        </p>
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-gray-50/80 border border-gray-100 flex items-start gap-3">
                      <span className="w-5 h-5 rounded-full bg-[#d2ecec] text-[#0D7F82] text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                        3
                      </span>
                      <div>
                        <p className="font-bold text-gray-800">Safety Verification & Export</p>
                        <p className="text-gray-500 mt-0.5">
                          Predictions pass through a 75% confidence safety gate and can be exported as structured clinical PDFs.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── Modal: Model Specifications ────────────────────────────────────── */}
      {showStatsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4 border border-gray-100 relative">
            <button
              onClick={() => setShowStatsModal(false)}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-700 p-1 rounded-lg"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[#EAF5F5] text-[#0D7F82] flex items-center justify-center">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-gray-800">Model Specifications</h3>
                <p className="text-[11px] text-gray-400">Architecture & Architecture Weights</p>
              </div>
            </div>

            <div className="space-y-2.5 text-xs text-gray-600">
              <div className="p-3 rounded-xl bg-gray-50 border border-gray-100 flex justify-between">
                <span className="font-semibold text-gray-700">Backbone Architecture:</span>
                <span className="font-mono text-[#0D7F82] font-bold">EfficientNet-B0 (timm)</span>
              </div>
              <div className="p-3 rounded-xl bg-gray-50 border border-gray-100 flex justify-between">
                <span className="font-semibold text-gray-700">Supported Target Classes:</span>
                <span className="font-mono font-bold">4 (Glioma, Meningioma, Pituitary, Healthy)</span>
              </div>
              <div className="p-3 rounded-xl bg-gray-50 border border-gray-100 flex justify-between">
                <span className="font-semibold text-gray-700">Input Resolution:</span>
                <span className="font-mono font-bold">224 x 224 x 3 (RGB Normalized)</span>
              </div>
              <div className="p-3 rounded-xl bg-gray-50 border border-gray-100 flex justify-between">
                <span className="font-semibold text-gray-700">Active Inference Hardware:</span>
                <span className="font-mono font-bold text-emerald-600">{backendStatus.device}</span>
              </div>
            </div>

            <button
              onClick={() => setShowStatsModal(false)}
              className="w-full py-2.5 bg-[#0D7F82] text-white rounded-xl text-xs font-bold hover:bg-[#0c7a7d] transition-colors"
            >
              Close Specifications
            </button>
          </div>
        </div>
      )}

      {/* ── Modal: Safety & Methodology ───────────────────────────────────── */}
      {showSafetyModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4 border border-gray-100 relative">
            <button
              onClick={() => setShowSafetyModal(false)}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-700 p-1 rounded-lg"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[#EAF5F5] text-[#0D7F82] flex items-center justify-center">
                <Shield className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-gray-800">Safety & Methodology</h3>
                <p className="text-[11px] text-gray-400">Confidence gates & oncology guidelines</p>
              </div>
            </div>

            <div className="space-y-3 text-xs text-gray-600 leading-relaxed">
              <div className="p-3.5 rounded-xl bg-[#EAF5F5] border border-[#0D7F82]/20 text-gray-700 space-y-1">
                <p className="font-bold text-[#0D7F82]">75% Confidence Safety Threshold</p>
                <p className="text-[11px]">
                  Any prediction with Softmax certainty below 75% is systematically flagged as <strong>INCONCLUSIVE</strong> to prevent false positives on noisy artifacts or non-cranial scans.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-gray-50 border border-gray-100 space-y-1">
                <p className="font-bold text-gray-800">Clinical Workflow Integration</p>
                <p className="text-[11px] text-gray-500">
                  This system is designed as a second-reader computer-aided detection (CAD) tool for neuroradiologists. It is not an autonomous diagnostic device.
                </p>
              </div>
            </div>

            <button
              onClick={() => setShowSafetyModal(false)}
              className="w-full py-2.5 bg-[#0D7F82] text-white rounded-xl text-xs font-bold hover:bg-[#0c7a7d] transition-colors"
            >
              Acknowledge Guidelines
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
