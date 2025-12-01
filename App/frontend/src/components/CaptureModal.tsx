import React, { useRef } from 'react';
import { Camera, Upload, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

interface CaptureModalProps {
  isOpen: boolean;
  onClose: () => void;
  onImageSelect: (imageData: string) => void;
}

const CaptureModal: React.FC<CaptureModalProps> = ({ isOpen, onClose, onImageSelect }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type and size
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        alert('Image file must be less than 10MB');
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const imageData = e.target?.result as string;
        onImageSelect(imageData);
        onClose();
      };
      reader.readAsDataURL(file);
    }
  };

  const handleCameraCapture = () => {
    cameraInputRef.current?.click();
  };

  const handleGalleryUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="vetra-container max-w-sm mx-auto">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="text-xl font-bold">Select Image</DialogTitle>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="w-5 h-5" />
            </Button>
          </div>
        </DialogHeader>

        <div className="space-y-4 p-6">
          <div className="text-center mb-6">
            <Camera className="w-16 h-16 mx-auto mb-4 text-primary" />
            <p className="text-muted-foreground">
              Capture or select a clear image of the livestock for accurate trait analysis
            </p>
          </div>

          <div className="space-y-4">
            <Button
              variant="vetra"
              size="vetra-primary" 
              className="w-full"
              onClick={handleCameraCapture}
            >
              <Camera className="w-6 h-6 mr-3" />
              <div className="text-left">
                <div className="font-bold">Capture Photo</div>
                <div className="text-sm opacity-90">Use device camera</div>
              </div>
            </Button>

            <Button
              variant="vetra-secondary"
              size="vetra-primary"
              className="w-full"
              onClick={handleGalleryUpload}
            >
              <Upload className="w-6 h-6 mr-3" />
              <div className="text-left">
                <div className="font-bold">Upload Image</div>
                <div className="text-sm opacity-90">Select from gallery</div>
              </div>
            </Button>
          </div>

          {/* Hidden file inputs */}
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
            capture="environment" // Use rear camera on mobile
          />
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />

          <div className="text-center mt-6">
            <p className="text-xs text-muted-foreground">
              Supported formats: JPG, PNG, WebP • Max size: 10MB
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default CaptureModal;