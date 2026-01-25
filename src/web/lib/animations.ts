/**
 * Shared animation configurations for OmniRank UI
 * Uses Framer Motion variants for consistent animations
 */

import type { Variants } from "framer-motion";

// Stagger children animation
export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

// Fade in from below
export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 24,
    },
  },
};

// Fade in from left
export const fadeInLeft: Variants = {
  hidden: { opacity: 0, x: -20 },
  show: {
    opacity: 1,
    x: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 24,
    },
  },
};

// Fade in from right
export const fadeInRight: Variants = {
  hidden: { opacity: 0, x: 20 },
  show: {
    opacity: 1,
    x: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 24,
    },
  },
};

// Scale up animation
export const scaleUp: Variants = {
  hidden: { opacity: 0, scale: 0.9 },
  show: {
    opacity: 1,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 20,
    },
  },
};

// Pulse animation for loading states
export const pulse: Variants = {
  hidden: { opacity: 0.5 },
  show: {
    opacity: 1,
    transition: {
      repeat: Infinity,
      repeatType: "reverse",
      duration: 0.8,
    },
  },
};

// Glow animation for accent elements
export const glow: Variants = {
  hidden: {
    boxShadow: "0 0 0 0 rgba(0, 240, 255, 0)",
  },
  show: {
    boxShadow: [
      "0 0 0 0 rgba(0, 240, 255, 0)",
      "0 0 20px 4px rgba(0, 240, 255, 0.3)",
      "0 0 0 0 rgba(0, 240, 255, 0)",
    ],
    transition: {
      repeat: Infinity,
      duration: 2,
    },
  },
};

// Page transition
export const pageTransition = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.3 },
};

// Card hover effect
export const cardHover = {
  scale: 1.02,
  transition: { type: "spring", stiffness: 400, damping: 17 },
};

// Button tap effect
export const buttonTap = {
  scale: 0.95,
};

// Smooth spring config for most animations
export const smoothSpring = {
  type: "spring" as const,
  stiffness: 300,
  damping: 30,
};

// Fast spring for micro-interactions
export const fastSpring = {
  type: "spring" as const,
  stiffness: 400,
  damping: 25,
};
