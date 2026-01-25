"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileSpreadsheet, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  onUpload: (file: File) => Promise<unknown>;
  isUploading?: boolean;
  isUploaded?: boolean;
  filename?: string | null;
  className?: string;
}

export function FileUpload({
  onUpload,
  isUploading = false,
  isUploaded = false,
  filename,
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
      if (!file.name.endsWith(".csv") && !file.name.endsWith(".json")) {
        setError("Please upload a CSV or JSON file");
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
      "application/json": [".json"],
    },
    maxFiles: 1,
    disabled: isUploading,
  });

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
                Supports CSV and JSON files with comparison data
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
