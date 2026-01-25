"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, MessageSquare, BarChart3, Network, Settings2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Home() {
  const [activeTab, setActiveTab] = useState("chat");

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background grid pattern */}
      <div className="fixed inset-0 grid-pattern opacity-50" />
      
      {/* Gradient overlay */}
      <div className="fixed inset-0 bg-gradient-to-br from-background via-background to-accent/5" />
      
      {/* Main content */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold gradient-text">OmniRank</h1>
              <p className="text-muted-foreground mt-1">
                Spectral Ranking Inference Platform
              </p>
            </div>
            <Button variant="outline" size="icon" className="glow-border">
              <Settings2 className="h-5 w-5" />
            </Button>
          </div>
        </motion.header>

        {/* Main workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left panel: Chat & Upload */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="lg:col-span-2"
          >
            <Card className="h-[calc(100vh-200px)] flex flex-col bg-card/80 backdrop-blur-sm glow-border">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-5 w-5 text-primary" />
                  <CardTitle className="text-lg">Analysis Console</CardTitle>
                </div>
                <CardDescription>
                  Upload your comparison data and interact with OmniRank
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                {/* Upload zone placeholder */}
                <div className="flex-1 flex items-center justify-center border-2 border-dashed border-border rounded-lg mb-4 hover:border-primary/50 transition-colors cursor-pointer group">
                  <div className="text-center p-8">
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary/10 mb-4 group-hover:bg-primary/20 transition-colors"
                    >
                      <Upload className="h-8 w-8 text-primary" />
                    </motion.div>
                    <p className="text-lg font-medium mb-1">
                      Drop your data here
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Supports CSV and JSON files with comparison data
                    </p>
                  </div>
                </div>

                {/* Chat input placeholder */}
                <div className="flex gap-2">
                  <div className="flex-1 bg-input rounded-lg px-4 py-3 text-muted-foreground">
                    Ask about your ranking analysis...
                  </div>
                  <Button className="glow-cyan">Send</Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Right panel: Results & Visualizations */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="h-[calc(100vh-200px)] bg-card/80 backdrop-blur-sm glow-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">Results</CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="rankings" className="w-full">
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

                  <TabsContent value="rankings" className="mt-0">
                    <div className="h-[400px] flex items-center justify-center text-muted-foreground border border-dashed border-border rounded-lg">
                      <div className="text-center">
                        <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-50" />
                        <p>Upload data to see rankings</p>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="heatmap" className="mt-0">
                    <div className="h-[400px] flex items-center justify-center text-muted-foreground border border-dashed border-border rounded-lg">
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
                  </TabsContent>

                  <TabsContent value="network" className="mt-0">
                    <div className="h-[400px] flex items-center justify-center text-muted-foreground border border-dashed border-border rounded-lg">
                      <div className="text-center">
                        <Network className="h-12 w-12 mx-auto mb-2 opacity-50" />
                        <p>Upload data to see network</p>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mt-8 text-center text-sm text-muted-foreground"
        >
          <p>
            Powered by Spectral Ranking Inference |{" "}
            <span className="text-primary">LLM Agent Platform</span>
          </p>
        </motion.footer>
      </div>
    </main>
  );
}
