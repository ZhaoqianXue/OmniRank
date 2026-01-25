"use client";

import { motion } from "framer-motion";
import { BarChart3, Network, Settings2, RotateCcw } from "lucide-react";
import { fadeInUp, fadeInLeft, fadeInRight, smoothSpring } from "@/lib/animations";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileUpload } from "@/components/upload/file-upload";
import { ChatInterface } from "@/components/chat/chat-interface";
import { ChatInput } from "@/components/chat/chat-input";
import { ConfigPanel } from "@/components/config/config-panel";
import { ProgressIndicator } from "@/components/ui/progress-indicator";
import { ErrorDisplay } from "@/components/ui/error-display";
import { RankingChart, HeatmapChart, NetworkGraph } from "@/components/visualizations";
import { useOmniRank } from "@/hooks/use-omnirank";

export default function Home() {
  const { state, handleUpload, startAnalysis, sendMessage, reset } = useOmniRank();

  const showConfig = state.status === "configuring" && state.schema;
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
            <Card className="h-[calc(100vh-80px)] flex flex-col bg-card/80 backdrop-blur-sm glow-border pt-4">


              <CardContent className="flex-1 flex flex-col min-h-0">
                {/* Upload zone - show when idle or to allow re-upload */}
                {(state.status === "idle" || state.status === "uploading") && (
                  <FileUpload
                    onUpload={handleUpload}
                    isUploading={state.status === "uploading"}
                    isUploaded={false}
                    filename={state.filename}
                    className="mb-4"
                  />
                )}

                {/* Config Panel - show when configuring */}
                {showConfig && state.schema && (
                  <div className="mb-4">
                    <ConfigPanel
                      schema={state.schema}
                      onStartAnalysis={startAnalysis}
                      isAnalyzing={state.status === "analyzing"}
                    />
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

                {/* Warnings display */}
                {state.warnings.length > 0 && state.status === "configuring" && (
                  <div className="mb-4 space-y-2">
                    {state.warnings.map((warning, idx) => (
                      <ErrorDisplay
                        key={idx}
                        message={warning.message}
                        type={warning.severity === "error" ? "warning" : "info"}
                      />
                    ))}
                  </div>
                )}

                {/* Results Visualization Tabs */}
                <div className="flex-1 min-h-0">
                  <Tabs defaultValue="rankings" className="w-full h-full flex flex-col">
                    <TabsList className="grid w-full grid-cols-3 mb-4">
                      <TabsTrigger value="rankings" className="text-xs">
                        <BarChart3 className="h-4 w-4 mr-1" />
                        Rankings
                      </TabsTrigger>
                      <TabsTrigger value="heatmap" className="text-xs">
                        <div className="h-4 w-4 mr-1 grid grid-cols-2 gap-0.5">
                          <div className="bg-current rounded-sm" />
                          <div className="bg-current/60 rounded-sm" />
                          <div className="bg-current/40 rounded-sm" />
                          <div className="bg-current/80 rounded-sm" />
                        </div>
                        Heatmap
                      </TabsTrigger>
                      <TabsTrigger value="network" className="text-xs">
                        <Network className="h-4 w-4 mr-1" />
                        Network
                      </TabsTrigger>
                    </TabsList>

                    <div className="flex-1 min-h-0">
                      <TabsContent value="rankings" className="mt-0 h-full">
                        {showResults && state.results ? (
                          <RankingChart items={state.results.items} className="h-full" />
                        ) : (
                          <div className="h-full flex items-center justify-center text-muted-foreground border border-dashed border-border rounded-lg">
                            <div className="text-center">
                              <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-50" />
                              <p>Upload data to see rankings</p>
                            </div>
                          </div>
                        )}
                      </TabsContent>

                      <TabsContent value="heatmap" className="mt-0 h-full">
                        {showResults && state.results ? (
                          <HeatmapChart results={state.results} className="h-full overflow-auto" />
                        ) : (
                          <div className="h-full flex items-center justify-center text-muted-foreground border border-dashed border-border rounded-lg">
                            <div className="text-center">
                              <div className="h-12 w-12 mx-auto mb-2 opacity-50 grid grid-cols-3 gap-1">
                                {[...Array(9)].map((_, i) => (
                                  <div
                                    key={i}
                                    className="bg-primary rounded-sm"
                                    style={{ opacity: Math.random() * 0.5 + 0.2 }}
                                  />
                                ))}
                              </div>
                              <p>Upload data to see heatmap</p>
                            </div>
                          </div>
                        )}
                      </TabsContent>

                      <TabsContent value="network" className="mt-0 h-full">
                        {showResults && state.results ? (
                          <NetworkGraph results={state.results} className="h-full" />
                        ) : (
                          <div className="h-full flex items-center justify-center text-muted-foreground border border-dashed border-border rounded-lg">
                            <div className="text-center">
                              <Network className="h-12 w-12 mx-auto mb-2 opacity-50" />
                              <p>Upload data to see network</p>
                            </div>
                          </div>
                        )}
                      </TabsContent>
                    </div>
                  </Tabs>
                </div>
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
                  className="h-full"
                />
              </CardContent>

              {/* Simple Chat Input */}
              <div className="p-2 border-t border-border/40">
                <ChatInput
                  onSend={sendMessage}
                  disabled={false}
                  placeholder="Type a message..."
                />
              </div>
            </Card>
          </motion.div>
        </div>


      </div>
    </main>
  );
}
