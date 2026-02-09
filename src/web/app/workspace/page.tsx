"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, History, House, LogIn, LogOut, Plus, Trophy } from "lucide-react";
import { fadeInLeft, fadeInRight } from "@/lib/animations";
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
import { cn } from "@/lib/utils";

export default function Home() {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

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

  const sidebarItems = [
    { id: "home", label: "Home", icon: House },
    { id: "new-chat", label: "New Chat", icon: Plus },
    { id: "history", label: "Chat History", icon: History },
    { id: "leaderboard", label: "LLM Leaderboard", icon: Trophy },
  ] as const;

  const handleSidebarAction = (menuId: string) => {
    if (menuId === "home") {
      window.open("/", "_blank", "noopener,noreferrer");
      return;
    }

    if (menuId === "leaderboard") {
      window.open("/leaderboard", "_blank", "noopener,noreferrer");
      return;
    }

    if (menuId === "new-chat") {
      reset();
    }
  };

  const handleLoginToggle = () => {
    setIsLoggedIn((prev) => !prev);
  };

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background grid pattern */}
      <div className="fixed inset-0 grid-pattern opacity-50" />

      {/* Gradient overlay */}
      <div className="fixed inset-0 bg-gradient-to-br from-background via-background to-accent/5" />

      <div className="relative z-10 flex min-h-screen">
        {/* Left sidebar */}
        <aside
          className={cn(
            "shrink-0 border-r border-border/40 bg-background backdrop-blur-sm flex flex-col justify-between transition-all duration-300 ease-in-out",
            isSidebarExpanded ? "w-52" : "w-12"
          )}
        >
          <div className="p-2">
            <Button
              variant="ghost"
              size="icon-sm"
              className="h-8 w-8"
              onClick={() => setIsSidebarExpanded((prev) => !prev)}
              aria-label={isSidebarExpanded ? "Collapse sidebar" : "Expand sidebar"}
            >
              {isSidebarExpanded ? (
                <ChevronLeft className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
            <div className="my-2 border-b border-border/40" />
            <nav className="space-y-1">
              {sidebarItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Button
                    key={item.id}
                    variant="ghost"
                    size="sm"
                    onClick={() => handleSidebarAction(item.id)}
                    className={cn(
                      "h-9 w-full justify-start px-2",
                      !isSidebarExpanded && "justify-center px-0"
                    )}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    {isSidebarExpanded && <span className="truncate">{item.label}</span>}
                  </Button>
                );
              })}
            </nav>
          </div>
          <div className="p-2 pt-0 border-t border-border/40">
            <Button
              variant="outline"
              onClick={handleLoginToggle}
              aria-label={isLoggedIn ? "Logout" : "Login"}
              className={cn(
                "mt-2 h-9 w-full justify-start gap-2 px-2",
                !isSidebarExpanded && "h-8 w-8 justify-center p-0"
              )}
            >
              <div
                className={cn(
                  "h-6 w-6 rounded-full border flex items-center justify-center text-[10px] font-semibold",
                  isLoggedIn
                    ? "border-primary/40 bg-primary text-primary-foreground"
                    : "border-border/80 bg-muted text-muted-foreground"
                )}
              >
                {isLoggedIn ? "U" : "?"}
              </div>
              {isSidebarExpanded && (
                <>
                  <span className="text-xs font-medium">
                    {isLoggedIn ? "Logout" : "Login"}
                  </span>
                  {isLoggedIn ? (
                    <LogOut className="h-3.5 w-3.5 ml-auto text-muted-foreground" />
                  ) : (
                    <LogIn className="h-3.5 w-3.5 ml-auto text-muted-foreground" />
                  )}
                </>
              )}
            </Button>
          </div>
        </aside>

        {/* Main workspace */}
        <div className="flex-1 min-w-0 px-4 pb-4 pt-4 md:px-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left panel: Canvas (Analysis Console) */}
            <motion.div
              variants={fadeInLeft}
              initial="hidden"
              animate="show"
              className="lg:col-span-2"
            >
              <Card className="h-[calc(100vh-32px)] flex flex-col bg-card backdrop-blur-sm glow-border py-4 relative overflow-hidden">
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
              <Card className="h-[calc(100vh-32px)] flex flex-col bg-card backdrop-blur-sm glow-border gap-0 p-0 overflow-hidden">
                {/* Chat Header */}
                <div className="flex items-center justify-center py-2 px-3 border-b border-border/40 min-h-[48px] shrink-0">
                  <div className="text-sm font-bold flex items-center justify-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                    OmniRank Agent
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
      </div>
    </main>
  );
}
