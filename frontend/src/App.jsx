import React, { useState } from 'react';
import axios from 'axios';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { Upload, FileText, Activity, CheckCircle, AlertCircle } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [patientName, setPatientName] = useState("");

  const processFile = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setResult(response.data);
    } catch (_error) {
      alert("Backend Error: Ensure main.py is running on port 8000.");
    }
    setLoading(false);
  };

  const downloadPDF = () => {
    try {
      const doc = new jsPDF();
      
      doc.setFontSize(22);
      doc.setTextColor(76, 29, 149);
      doc.text("Brain Tumor Detection Report", 20, 20);
      
      doc.setFontSize(10);
      doc.setTextColor(100, 116, 139);
      doc.text(`Patient Name: ${patientName || "Not Specified"}`, 20, 30);
      doc.text(`Report Date: ${new Date().toLocaleString()}`, 20, 35);
      
      // Using the direct autoTable function
      autoTable(doc, {
        startY: 45,
        head: [['Clinical Metric', 'Analysis Result']],
        body: [
          ['Diagnosis Type', result.diagnosis],
          ['Detection Confidence', `${result.confidence}%`],
          ['Scan Classification', result.status],
          ['AI Model Architecture', 'YOLOv8 + EfficientNet-B0'],
        ],
        headStyles: { fillColor: [76, 29, 149] },
        styles: { fontSize: 11 }
      });

      const finalY = doc.lastAutoTable.finalY || 90;
      doc.setFontSize(10);
      doc.setTextColor(150, 150, 150);
      doc.text("Disclaimer: Automated AI output. Please consult a radiologist.", 20, finalY + 20);

      doc.save(`Tumor_Report_${result.diagnosis}.pdf`);
    } catch (err) {
      console.error(err);
      alert("PDF Error. Check if you installed jspdf-autotable correctly.");
    }
  };

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', backgroundColor: 'white', minHeight: '100vh', width: '100%' }}>
      <header style={{ textAlign: 'center', marginBottom: '60px' }}>
        <h1 style={{ color: '#4c1d95', marginBottom: '10px', fontSize: '2.8rem' }}>🧠 Tumor Detector</h1>
        <p style={{ color: '#64748b', fontSize: '1.2rem' }}>AI-Powered MRI Analysis System</p>
      </header>
      
      <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', gap: '30px' }}>
        <div style={{ flex: 1, background: 'white', padding: '30px', borderRadius: '24px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.05)', border: '1px solid #f1f5f9' }}>
          <h3 style={{ marginBottom: '20px' }}>MRI Input</h3>
          <input 
            type="text" 
            placeholder="Enter Patient Name..." 
            value={patientName} 
            onChange={(e) => setPatientName(e.target.value)}
            style={{ width: '100%', padding: '10px', marginBottom: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}
          />
          <div 
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); processFile(e.dataTransfer.files[0]); }}
            style={{ border: `2px dashed ${isDragging ? '#7c3aed' : '#e2e8f0'}`, borderRadius: '20px', padding: '20px', textAlign: 'center', minHeight: '220px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: isDragging ? '#f5f3ff' : '#fafafa' }}
          >
            {preview ? <img src={preview} alt="MRI" style={{ maxWidth: '100%', borderRadius: '12px' }} /> : <div style={{ color: '#94a3b8' }}><Upload size={48} /><p>Drop MRI Scan Here</p></div>}
          </div>
          <input type="file" onChange={(e) => processFile(e.target.files[0])} style={{ marginTop: '25px', display: 'block', width: '100%' }} />
          <button onClick={handleUpload} disabled={loading || !file} style={{ width: '100%', marginTop: '25px', padding: '16px', background: '#7c3aed', color: 'white', border: 'none', borderRadius: '12px', cursor: 'pointer', fontWeight: 'bold' }}>
            {loading ? "Analyzing..." : "Run Detection"}
          </button>
        </div>

        <div style={{ flex: 1, background: 'white', padding: '30px', borderRadius: '24px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.05)', border: '1px solid #f1f5f9' }}>
          <h3 style={{ marginBottom: '25px' }}>Diagnostic Output</h3>
          {result ? (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', padding: '20px', borderRadius: '16px', backgroundColor: result.status === "Healthy" ? '#f0fdf4' : '#fef2f2', border: `1px solid ${result.status === "Healthy" ? '#bbf7d0' : '#fecaca'}`, marginBottom: '25px' }}>
                {result.status === "Healthy" ? <CheckCircle color="#16a34a" size={28} /> : <AlertCircle color="#dc2626" size={28} />}
                <span style={{ fontSize: '1.6rem', fontWeight: 'bold', color: result.status === "Healthy" ? '#166534' : '#991b1b' }}>{result.diagnosis}</span>
              </div>
              <p>System Confidence: <strong>{result.confidence}%</strong></p>
              <button onClick={downloadPDF} style={{ width: '100%', marginTop: '30px', padding: '12px', border: '2px solid #7c3aed', color: '#7c3aed', background: 'white', borderRadius: '12px', fontWeight: 'bold', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                <FileText size={20} /> Download Report PDF
              </button>
            </div>
          ) : <div style={{ textAlign: 'center', marginTop: '80px', color: '#cbd5e1' }}><Activity size={64} /><p>Waiting for scan...</p></div>}
        </div>
      </div>
    </div>
  );
}
export default App;
