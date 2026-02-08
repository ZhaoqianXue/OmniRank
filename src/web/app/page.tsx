"use client";

import { motion } from "framer-motion";
import { Settings2, RotateCcw } from "lucide-react";
import { fadeInUp, fadeInLeft, fadeInRight } from "@/lib/animations";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FileUpload } from "@/components/upload/file-upload";
import { ExampleDataSelector } from "@/components/upload/example-data-selector";
import { DataPreviewComponent } from "@/components/upload/data-preview";
import { ChatInterface } from "@/components/chat/chat-interface";
import { ChatInput } from "@/components/chat/chat-input";
import { ProgressIndicator } from "@/components/ui/progress-indicator";
import { ErrorDisplay } from "@/components/ui/error-display";
import { ReportOverlay } from "@/components/report";
import { useOmniRank } from "@/hooks/use-omnirank";

export default function Home() {
  const {
    state,
    handleUpload,
    loadExampleData,
    cancelData,
    startAnalysis,
    sendMessage,
    reset,
    toggleReportVisibility,
    hideReport,
    exampleDatasets,
  } = useOmniRank();

  const isIdle = state.status === "idle";
  const isUploading = state.status === "uploading";
  const isPreviewLoading = isUploading && !state.dataPreview;
  // Show sticker mode as soon as uploading starts (filename is set)
  // or when in configuring/analyzing/completed state
  const hasData = state.filename && (state.status === "uploading" || state.status === "configuring" || state.status === "analyzing" || state.status === "completed");
  const isAnalyzing = state.status === "analyzing";
  const showProgress = state.status === "analyzing";
  const showResults = state.status === "completed" && state.results;

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background grid pattern */}
      <div className="fixed inset-0 grid-pattern opacity-50" />

      {/* Gradient overlay */}
      <div className="fixed inset-0 bg-gradient-to-br from-background via-background to-accent/5" />

      {/* Main content */}
      <div className="relative z-10 w-full px-6 pb-4 pt-2">
        {/* Header - Compact Navbar */}
        <motion.header
          variants={fadeInUp}
          initial="hidden"
          animate="show"
          className="mb-3 border-b border-border/40 pb-1"
        >
          <div className="flex items-center justify-between h-9">
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold gradient-text">OmniRank</h1>
            </div>
            <div className="flex items-center gap-2">
              {state.sessionId && (
                <Button variant="ghost" size="sm" onClick={reset} className="h-7 text-xs">
                  <RotateCcw className="h-3 w-3 mr-1.5" />
                  New Session
                </Button>
              )}
              <Button variant="ghost" size="icon" className="h-7 w-7 hover:bg-accent/50">
                <Settings2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        </motion.header>

        {/* Main workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left panel: Canvas (Analysis Console) */}
          <motion.div
            variants={fadeInLeft}
            initial="hidden"
            animate="show"
            className="lg:col-span-2"
          >
            <Card className="h-[calc(100vh-80px)] flex flex-col bg-card/80 backdrop-blur-sm glow-border py-4 relative overflow-hidden">
              {/* Report Overlay - covers the entire card when visible */}
              {showResults && (
                <ReportOverlay
                  isVisible={state.isReportVisible}
                  results={state.results}
                  reportOutput={state.reportOutput}
                  plots={state.plots}
                  artifacts={state.artifacts}
                  sessionId={state.sessionId}
                  schema={state.schema}
                  config={state.config}
                  onClose={hideReport}
                  onSendMessage={sendMessage}
                />
              )}

              <CardContent className="flex-1 flex flex-col min-h-0">
                {/* Initial State: Upload zone + Example selector - only show when truly idle */}
                {isIdle && !hasData && (
                  <div className="space-y-4 mb-4">
                    <FileUpload
                      onUpload={handleUpload}
                      mode="dropzone"
                      isUploading={false}
                      isUploaded={false}
                      filename={null}
                    />
                    <ExampleDataSelector
                      examples={exampleDatasets}
                      onSelect={loadExampleData}
                      disabled={false}
                    />
                  </div>
                )}

                {/* Data Loaded State: Sticker + Preview - shows immediately when file is selected */}
                {hasData && (
                  <div className="space-y-4 flex-1 flex flex-col min-h-0">
                    <FileUpload
                      onUpload={handleUpload}
                      onCancel={!isUploading ? cancelData : undefined}  // Disable cancel during upload
                      mode="sticker"
                      filename={state.filename}
                      isExample={state.dataSource === "example"}
                      isUploading={isUploading}  // Pass uploading state for visual feedback
                    />
                    <div className="flex-1 min-h-0">
                      <DataPreviewComponent
                        preview={state.dataPreview}
                        exampleInfo={state.exampleDataInfo}
                        isLoading={isPreviewLoading}
                        className="h-full"
                      />
                    </div>
                  </div>
                )}

                {/* Progress indicator */}
                {showProgress && (
                  <div className="mb-4">
                    <ProgressIndicator
                      progress={state.progress}
                      message={state.progressMessage}
                    />
                  </div>
                )}

                {/* Error display */}
                {state.status === "error" && state.error && (
                  <div className="mb-4">
                    <ErrorDisplay
                      title="Analysis Error"
                      message={state.error}
                      type="error"
                      onRetry={reset}
                    />
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Right panel: Chat Panel */}
          <motion.div
            variants={fadeInRight}
            initial="hidden"
            animate="show"
          >
            <Card className="h-[calc(100vh-80px)] flex flex-col bg-card/80 backdrop-blur-sm glow-border gap-0 p-0 overflow-hidden">
              {/* Chat Header */}
              <div className="flex items-center justify-center py-2 px-3 border-b border-border/40 min-h-[48px] shrink-0">
                <div className="text-sm font-medium flex items-center justify-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                  OmniRank Assistant
                </div>
              </div>

              {/* Chat Messages */}
              <CardContent className="flex-1 min-h-0 p-0">
                <ChatInterface
                  messages={state.messages}
                  onStartAnalysis={startAnalysis}
                  onSendMessage={sendMessage}
                  isAnalyzing={isAnalyzing}
                  isCompleted={!!showResults}
                  isReportVisible={state.isReportVisible}
                  onToggleReport={toggleReportVisibility}
                  className="h-full"
                />
              </CardContent>

              {/* Smart Chat Input with Quick Start */}
              <div className="p-2 border-t border-border/40">
                <ChatInput
                  onSend={sendMessage}
                  disabled={false}
                  placeholder="Type your message..."
                  status={state.status}
                  schema={state.schema}
                  results={state.results}
                />
              </div>
            </Card>
          </motion.div>
        </div>


      </div>
    </main>
  );
}
