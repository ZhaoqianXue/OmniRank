"use client";

import { useEffect } from "react";

// Ensure Material Symbols font is loaded
export function MaterialSymbol({ 
  icon, 
  className = "", 
  style = {},
  ...props 
}: { 
  icon: string; 
  className?: string;
  style?: React.CSSProperties;
  [key: string]: any;
}) {
  useEffect(() => {
    // Dynamically load Material Symbols font if not already loaded
    const linkId = "material-symbols-font";
    if (!document.getElementById(linkId)) {
      const link = document.createElement("link");
      link.id = linkId;
      link.rel = "stylesheet";
      link.href = "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap";
      document.head.appendChild(link);
    }
  }, []);

  return (
    <span
      className={`material-symbols-outlined ${className}`}
      style={{
        fontFamily: '"Material Symbols Outlined"',
        fontWeight: "normal",
        fontStyle: "normal",
        fontSize: "24px",
        lineHeight: "1",
        letterSpacing: "normal",
        textTransform: "none",
        display: "inline-block",
        whiteSpace: "nowrap",
        wordWrap: "normal",
        direction: "ltr",
        WebkitFontFeatureSettings: '"liga"',
        fontFeatureSettings: '"liga"',
        WebkitFontSmoothing: "antialiased",
        fontVariationSettings: '"FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24',
        ...style,
      }}
      {...props}
    >
      {icon}
    </span>
  );
}
