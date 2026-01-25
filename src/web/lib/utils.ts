import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format a number with appropriate precision.
 */
export function formatNumber(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

/**
 * Generate a random ID.
 */
export function generateId(): string {
  return Math.random().toString(36).substring(2, 11);
}

/**
 * Delay execution for a specified time.
 */
export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Format file size in human-readable format.
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

/**
 * Check if a file is a valid data file (CSV or JSON).
 */
export function isValidDataFile(file: File): boolean {
  const validTypes = [
    "text/csv",
    "application/json",
    "application/vnd.ms-excel",
  ];
  const validExtensions = [".csv", ".json"];
  
  const hasValidType = validTypes.includes(file.type);
  const hasValidExtension = validExtensions.some((ext) =>
    file.name.toLowerCase().endsWith(ext)
  );
  
  return hasValidType || hasValidExtension;
}
