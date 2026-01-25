"use client";

import { useState, useMemo } from "react";
import { Info, Table2, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import type { DataPreview, ExampleDataInfo } from "@/lib/api";

interface DataPreviewProps {
  preview: DataPreview | null;
  exampleInfo?: ExampleDataInfo | null;
  isLoading?: boolean;
  className?: string;
}

const ROWS_PER_PAGE = 100;

export function DataPreviewComponent({
  preview,
  exampleInfo,
  isLoading = false,
  className,
}: DataPreviewProps) {
  const [currentPage, setCurrentPage] = useState(1);

  // Calculate pagination
  const { paginatedRows, totalPages, startRow, endRow } = useMemo(() => {
    if (!preview) {
      return { paginatedRows: [], totalPages: 0, startRow: 0, endRow: 0 };
    }

    const total = preview.rows.length;
    const pages = Math.ceil(total / ROWS_PER_PAGE);
    const start = (currentPage - 1) * ROWS_PER_PAGE;
    const end = Math.min(start + ROWS_PER_PAGE, total);
    const rows = preview.rows.slice(start, end);

    return {
      paginatedRows: rows,
      totalPages: pages,
      startRow: start + 1,
      endRow: end,
    };
  }, [preview, currentPage]);

  // Reset page when preview changes
  useMemo(() => {
    setCurrentPage(1);
  }, [preview?.totalRows]);

  if (isLoading) {
    return (
      <Card className={cn("", className)}>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Table2 className="h-4 w-4" />
            Data Preview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!preview) {
    return (
      <Card className={cn("", className)}>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Table2 className="h-4 w-4" />
            Data Preview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-24 text-muted-foreground text-sm">
            No data to preview
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("flex flex-col", className)}>
      <CardHeader className="pb-0 shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <Table2 className="h-4 w-4" />
            Data Preview
          </CardTitle>
          <Badge variant="secondary" className="text-xs">
            {preview.totalRows.toLocaleString()} rows
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 pt-0">
        {/* Example data description */}
        {exampleInfo && (
          <div className="flex gap-2 p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 mb-3 shrink-0 min-h-[72px] -mt-4">
            <Info className="h-4 w-4 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
              {exampleInfo.description}
            </p>
          </div>
        )}

        {/* Data table preview - fills remaining space */}
        <ScrollArea className="flex-1 min-h-0 rounded-md border">
          <div className="min-w-max">
            <table className="w-full text-xs">
              <thead className="bg-muted sticky top-0 z-10">
                <tr>
                  {preview.columns.map((col, i) => (
                    <th
                      key={i}
                      className="px-3 py-2 text-left font-medium text-muted-foreground border-b whitespace-nowrap bg-muted"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {paginatedRows.map((row, rowIdx) => (
                  <tr
                    key={rowIdx}
                    className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                  >
                    {preview.columns.map((col, colIdx) => {
                      const value = row[col];
                      const displayValue =
                        typeof value === "number"
                          ? parseFloat(value.toFixed(3)).toString()
                          : String(value ?? "");
                      const truncated =
                        displayValue.length > 25
                          ? displayValue.substring(0, 25) + "..."
                          : displayValue;

                      return (
                        <td
                          key={colIdx}
                          className="px-3 py-1.5 whitespace-nowrap"
                          title={displayValue}
                        >
                          {truncated}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ScrollArea>

        {/* Pagination controls */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between pt-3 shrink-0">
            <p className="text-xs text-muted-foreground">
              Showing {startRow.toLocaleString()}-{endRow.toLocaleString()} of {preview.totalRows.toLocaleString()} rows
            </p>
            <div className="flex items-center gap-1">
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => setCurrentPage(1)}
                disabled={currentPage === 1}
              >
                <ChevronsLeft className="h-3 w-3" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="h-3 w-3" />
              </Button>
              <span className="text-xs px-2 min-w-[60px] text-center">
                {currentPage} / {totalPages}
              </span>
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
              >
                <ChevronRight className="h-3 w-3" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => setCurrentPage(totalPages)}
                disabled={currentPage === totalPages}
              >
                <ChevronsRight className="h-3 w-3" />
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
