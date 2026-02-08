"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileSpreadsheet, AlertCircle, CheckCircle2, Loader2, X, Database } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

type UploadMode = "dropzone" | "sticker";

interface FileUploadProps {
  onUpload: (file: File) => Promise<unknown>;
  onCancel?: () => void;
  mode?: UploadMode;
  isUploading?: boolean;
  isUploaded?: boolean;
  filename?: string | null;
  isExample?: boolean;
  className?: string;
}

export function FileUpload({
  onUpload,
  onCancel,
  mode = "dropzone",
  isUploading = false,
  isUploaded = false,
  filename,
  isExample = false,
  className,
}: FileUploadProps) {
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      setError(null);
      
      if (acceptedFiles.length === 0) {
        return;
      }

      const file = acceptedFiles[0];
      
      // Validate file type
      if (!file.name.endsWith(".csv")) {
        setError("Please upload a CSV file");
        return;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError("File size must be less than 10MB");
        return;
      }

      try {
        await onUpload(file);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed");
      }
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
    },
    maxFiles: 1,
    disabled: isUploading || mode === "sticker",
  });

  // Sticker mode - compact display with cancel button
  if (mode === "sticker" && filename) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className={cn("relative", className)}
      >
        <div className={cn(
          "flex items-center gap-3 px-4 py-3 rounded-lg border",
          isUploading 
            ? "bg-gradient-to-r from-blue-500/10 to-indigo-500/10 border-blue-500/30"
            : "bg-gradient-to-r from-green-500/10 to-emerald-500/10 border-green-500/30"
        )}>
          <div className={cn(
            "flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center",
            isUploading ? "bg-blue-500/20" : "bg-green-500/20"
          )}>
            {isUploading ? (
              <Loader2 className="h-4 w-4 text-blue-600 dark:text-blue-400 animate-spin" />
            ) : isExample ? (
              <Database className="h-4 w-4 text-green-600 dark:text-green-400" />
            ) : (
              <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{filename}</p>
            <p className="text-xs text-muted-foreground">
              {isUploading 
                ? "Processing with Data Agent..." 
                : isExample 
                  ? "Example dataset loaded" 
                  : "File uploaded successfully"}
            </p>
          </div>
          {onCancel && !isUploading && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 hover:bg-destructive/10 hover:text-destructive"
              onClick={onCancel}
            >
              <X className="h-4 w-4" />
              <span className="sr-only">Remove file</span>
            </Button>
          )}
        </div>
      </motion.div>
    );
  }

  // Dropzone mode - standard upload interface
  return (
    <div className={cn("relative", className)}>
      <div
        {...getRootProps()}
        className={cn(
          "relative flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-lg transition-all cursor-pointer",
          isDragActive && "border-primary bg-primary/5",
          isUploading && "cursor-wait opacity-70",
          isUploaded && "border-green-500/50 bg-green-500/5",
          error && "border-destructive/50 bg-destructive/5",
          !isDragActive && !isUploaded && !error && "border-border hover:border-primary/50 hover:bg-primary/5"
        )}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {isUploading ? (
            <motion.div
              key="uploading"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-center"
            >
              <Loader2 className="h-12 w-12 mx-auto mb-4 text-primary animate-spin" />
              <p className="text-lg font-medium">Uploading...</p>
              <p className="text-sm text-muted-foreground">Please wait</p>
            </motion.div>
          ) : isUploaded && filename ? (
            <motion.div
              key="uploaded"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-center"
            >
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-green-500/10 mb-4">
                <CheckCircle2 className="h-6 w-6 text-green-500" />
              </div>
              <div className="flex items-center gap-2 justify-center mb-2">
                <FileSpreadsheet className="h-5 w-5 text-muted-foreground" />
                <p className="text-lg font-medium">{filename}</p>
              </div>
              <p className="text-sm text-muted-foreground">
                Click or drag to replace
              </p>
            </motion.div>
          ) : error ? (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-center"
            >
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-destructive/10 mb-4">
                <AlertCircle className="h-6 w-6 text-destructive" />
              </div>
              <p className="text-lg font-medium text-destructive">{error}</p>
              <p className="text-sm text-muted-foreground mt-1">
                Click or drag to try again
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="default"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-center"
            >
              <motion.div
                animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
                className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4"
              >
                <Upload className={cn("h-8 w-8 text-primary", isDragActive && "text-primary")} />
              </motion.div>
              <p className="text-lg font-medium mb-1">
                {isDragActive ? "Drop your file here" : "Drop your data here"}
              </p>
              <p className="text-sm text-muted-foreground">
                Supports CSV files with comparison data
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
