import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Loader2, CheckCircle, AlertCircle, RotateCcw, Download, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import { processImage, type TraitExtractionResult } from '@/lib/api';
import CaptureModal from './CaptureModal';
import { z } from 'zod';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

// Input validation schema
const imageProcessingSchema = z.object({
  imageData: z.string().min(1, 'Image data is required'),
  animalType: z.string().max(50, 'Animal type must be less than 50 characters').optional(),
  notes: z.string().max(500, 'Notes must be less than 500 characters').optional(),
});

const TraitExtraction = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<TraitExtractionResult | null>(null);
  const [showCaptureModal, setShowCaptureModal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const handleImageSelect = (imageData: string) => {
    try {
      imageProcessingSchema.parse({ imageData });
      setSelectedImage(imageData);
      setError(null);
    } catch (err) {
      if (err instanceof z.ZodError) {
        setError(err.issues[0].message);
        toast({
          title: "Invalid Image",
          description: err.issues[0].message,
          variant: "destructive",
        });
      }
    }
  };

  const processImageAnalysis = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setError(null);

    try {
      const validatedData = imageProcessingSchema.parse({
        imageData: selectedImage,
        animalType: 'cattle',
        notes: 'Processed via VETRA web app'
      });

      const response = await processImage(
        validatedData.imageData,
        validatedData.animalType,
        validatedData.notes
      );

      if (response.success && response.data) {
        setResults(response.data);
        toast({
          title: "Analysis Complete",
          description: `Trait extraction successful with ${response.data.confidence}% confidence`,
        });
      } else {
        throw new Error(response.error || 'Failed to process image');
      }
    } catch (error) {
      console.error('Processing error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Processing failed';
      setError(errorMessage);
      toast({
        title: "Processing Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const resetProcess = () => {
    setSelectedImage(null);
    setResults(null);
    setIsProcessing(false);
    setError(null);
  };

  const startNewAnalysis = () => {
    setShowCaptureModal(true);
  };

  const getBodyScoreLabel = (score: number): string => {
    if (score <= 3) return 'Weak';
    if (score <= 6) return 'Average';
    return 'Strong';
  };

  const downloadPDF = () => {
    if (!results) return;

    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    
    // Header
    doc.setFontSize(20);
    doc.setTextColor(34, 139, 34); // Forest green
    doc.text('VETRA - Trait Analysis Report', pageWidth / 2, 20, { align: 'center' });
    
    // Date and ID
    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(`Report ID: ${results.id}`, 14, 35);
    doc.text(`Date: ${new Date(results.timestamp).toLocaleString()}`, 14, 42);
    doc.text(`Animal Type: ${results.animalType || 'Cattle'}`, 14, 49);
    
    // Summary section
    doc.setFontSize(14);
    doc.setTextColor(0);
    doc.text('Analysis Summary', 14, 62);
    
    // Main measurements table
    autoTable(doc, {
      startY: 68,
      head: [['Parameter', 'Value', 'Unit']],
      body: [
        ['Body Length', results.bodyLength.toString(), 'cm'],
        ['Withers Height', results.withersHeight.toString(), 'cm'],
        ['Girth', results.girth.toString(), 'cm'],
        ['Body Score', `${results.bodyScore} (${getBodyScoreLabel(results.bodyScore)})`, '1-9 Scale'],
        ['Estimated Weight', results.estimatedWeight?.toString() || 'N/A', 'kg'],
        ['Analysis Confidence', results.confidence.toString(), '%'],
      ],
      theme: 'grid',
      headStyles: { fillColor: [34, 139, 34], textColor: 255 },
      alternateRowStyles: { fillColor: [240, 255, 240] },
      styles: { fontSize: 11, cellPadding: 5 },
    });

    // Body Score interpretation
    const finalY = (doc as any).lastAutoTable.finalY || 150;
    doc.setFontSize(12);
    doc.setTextColor(0);
    doc.text('Body Score Interpretation:', 14, finalY + 15);
    
    autoTable(doc, {
      startY: finalY + 20,
      head: [['Score Range', 'Condition', 'Description']],
      body: [
        ['1 - 3', 'Weak', 'Animal needs immediate attention'],
        ['4 - 6', 'Average', 'Acceptable condition'],
        ['7 - 9', 'Strong', 'Excellent health and condition'],
      ],
      theme: 'striped',
      headStyles: { fillColor: [70, 130, 180] },
      styles: { fontSize: 9, cellPadding: 3 },
    });

    // Footer
    const footerY = doc.internal.pageSize.getHeight() - 15;
    doc.setFontSize(8);
    doc.setTextColor(150);
    doc.text('Generated by VETRA - AI-Powered Livestock Trait Analysis', pageWidth / 2, footerY, { align: 'center' });
    
    doc.save(`VETRA_Report_${results.id}.pdf`);
    
    toast({
      title: "PDF Downloaded",
      description: "Your analysis report has been saved",
    });
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="flex items-center gap-4 p-4 md:p-6 border-b border-border bg-card">
        <Link to="/">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </Link>
        <h1 className="text-xl md:text-2xl font-bold text-foreground">Trait Extraction</h1>
      </header>

      <div className="p-4 md:p-8 lg:p-12 max-w-6xl mx-auto">
        {!selectedImage && !results && (
          <Card className="max-w-2xl mx-auto">
            <CardContent className="pt-8 pb-8 text-center">
              <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-primary to-primary/70 rounded-full flex items-center justify-center shadow-lg">
                <span className="text-3xl font-bold text-primary-foreground">AI</span>
              </div>
              <h2 className="text-2xl font-semibold mb-3 text-foreground">AI-Powered Trait Analysis</h2>
              <p className="text-muted-foreground mb-8 text-lg">
                Upload or capture livestock images for instant body measurements and health scoring
              </p>
              
              <Button 
                variant="vetra" 
                size="vetra-primary" 
                className="w-full max-w-sm mx-auto"
                onClick={startNewAnalysis}
              >
                Start Analysis
              </Button>
            </CardContent>
          </Card>
        )}

        {selectedImage && !results && (
          <Card className="max-w-3xl mx-auto">
            <CardContent className="pt-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <img 
                    src={selectedImage} 
                    alt="Selected livestock" 
                    className="w-full rounded-lg max-h-80 object-cover border border-border"
                  />
                </div>
                
                <div className="flex flex-col justify-center space-y-4">
                  <h3 className="text-lg font-semibold text-foreground">Image Selected</h3>
                  <p className="text-muted-foreground">
                    Click "Process Image" to analyze the livestock traits using AI.
                  </p>
                  
                  {error && (
                    <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-destructive" />
                      <p className="text-sm text-destructive">{error}</p>
                    </div>
                  )}
                  
                  <div className="flex gap-3 pt-4">
                    <Button 
                      variant="vetra" 
                      size="lg" 
                      className="flex-1"
                      onClick={processImageAnalysis}
                      disabled={isProcessing}
                    >
                      {isProcessing ? (
                        <>
                          <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        'Process Image'
                      )}
                    </Button>
                    
                    <Button 
                      variant="outline" 
                      size="lg"
                      onClick={resetProcess}
                      disabled={isProcessing}
                    >
                      <RotateCcw className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {results && (
          <div className="space-y-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <CardTitle className="text-xl md:text-2xl">Analysis Complete</CardTitle>
                </div>
                <Button 
                  variant="outline" 
                  onClick={downloadPDF}
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  <span className="hidden sm:inline">Download PDF</span>
                </Button>
              </CardHeader>
              <CardContent>
                {/* Summary Cards */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
                  <div className="bg-gradient-to-br from-primary/10 to-primary/5 rounded-xl p-4 text-center border border-primary/20">
                    <div className="text-2xl md:text-3xl font-bold text-primary mb-1">{results.bodyLength}</div>
                    <div className="text-xs md:text-sm text-muted-foreground">Body Length (cm)</div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-blue-500/10 to-blue-500/5 rounded-xl p-4 text-center border border-blue-500/20">
                    <div className="text-2xl md:text-3xl font-bold text-blue-600 mb-1">{results.withersHeight}</div>
                    <div className="text-xs md:text-sm text-muted-foreground">Height (cm)</div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-purple-500/10 to-purple-500/5 rounded-xl p-4 text-center border border-purple-500/20">
                    <div className="text-2xl md:text-3xl font-bold text-purple-600 mb-1">{results.girth}</div>
                    <div className="text-xs md:text-sm text-muted-foreground">Girth (cm)</div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-orange-500/10 to-orange-500/5 rounded-xl p-4 text-center border border-orange-500/20">
                    <div className="text-2xl md:text-3xl font-bold text-orange-600 mb-1">{results.bodyScore}</div>
                    <div className="text-xs md:text-sm text-muted-foreground">Body Score</div>
                  </div>
                  
                  <div className="bg-gradient-to-br from-green-500/10 to-green-500/5 rounded-xl p-4 text-center border border-green-500/20 col-span-2 md:col-span-1">
                    <div className="text-2xl md:text-3xl font-bold text-green-600 mb-1">{results.confidence}%</div>
                    <div className="text-xs md:text-sm text-muted-foreground">Confidence</div>
                  </div>
                </div>

                {/* Detailed Results Table */}
                <div className="rounded-lg border border-border overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/50">
                        <TableHead className="font-semibold">Parameter</TableHead>
                        <TableHead className="font-semibold">Value</TableHead>
                        <TableHead className="font-semibold">Unit</TableHead>
                        <TableHead className="font-semibold hidden md:table-cell">Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableRow>
                        <TableCell className="font-medium">Body Length</TableCell>
                        <TableCell>{results.bodyLength}</TableCell>
                        <TableCell>cm</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs">Normal</span>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Withers Height</TableCell>
                        <TableCell>{results.withersHeight}</TableCell>
                        <TableCell>cm</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs">Normal</span>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Girth</TableCell>
                        <TableCell>{results.girth}</TableCell>
                        <TableCell>cm</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs">Normal</span>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Body Score</TableCell>
                        <TableCell>{results.bodyScore}/9 ({getBodyScoreLabel(results.bodyScore)})</TableCell>
                        <TableCell>1-9 Scale</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className={`px-2 py-1 rounded-full text-xs ${
                            results.bodyScore >= 7 ? 'bg-green-100 text-green-700' : 
                            results.bodyScore >= 4 ? 'bg-yellow-100 text-yellow-700' : 
                            'bg-red-100 text-red-700'
                          }`}>
                            {getBodyScoreLabel(results.bodyScore)}
                          </span>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Estimated Weight</TableCell>
                        <TableCell>{results.estimatedWeight || 'N/A'}</TableCell>
                        <TableCell>kg</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded-full text-xs">Calculated</span>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Analysis Confidence</TableCell>
                        <TableCell>{results.confidence}</TableCell>
                        <TableCell>%</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className={`px-2 py-1 rounded-full text-xs ${
                            results.confidence >= 90 ? 'bg-green-100 text-green-700' : 
                            results.confidence >= 80 ? 'bg-yellow-100 text-yellow-700' : 
                            'bg-red-100 text-red-700'
                          }`}>
                            {results.confidence >= 90 ? 'High' : results.confidence >= 80 ? 'Medium' : 'Low'}
                          </span>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Animal Type</TableCell>
                        <TableCell className="capitalize">{results.animalType || 'Cattle'}</TableCell>
                        <TableCell>-</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs">Detected</span>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Report ID</TableCell>
                        <TableCell className="font-mono text-xs">{results.id}</TableCell>
                        <TableCell>-</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs">System</span>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Timestamp</TableCell>
                        <TableCell>{new Date(results.timestamp).toLocaleString()}</TableCell>
                        <TableCell>-</TableCell>
                        <TableCell className="hidden md:table-cell">
                          <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs">Recorded</span>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-3 mt-6">
                  <Button 
                    variant="vetra" 
                    size="lg" 
                    className="flex-1"
                    onClick={downloadPDF}
                  >
                    <FileText className="w-5 h-5 mr-2" />
                    Download Full Report
                  </Button>
                  
                  <Link to="/data-records" className="flex-1">
                    <Button variant="outline" size="lg" className="w-full">
                      View All Records
                    </Button>
                  </Link>
                  
                  <Button 
                    variant="outline" 
                    size="lg"
                    onClick={startNewAnalysis}
                  >
                    New Analysis
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      <CaptureModal
        isOpen={showCaptureModal}
        onClose={() => setShowCaptureModal(false)}
        onImageSelect={handleImageSelect}
      />
    </div>
  );
};

export default TraitExtraction;