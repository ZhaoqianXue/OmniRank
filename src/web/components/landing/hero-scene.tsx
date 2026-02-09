"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { cn } from "@/lib/utils";

interface HeroSceneProps {
  className?: string;
}

interface FlowParticle {
  x: number;
  y: number;
  speed: number;
  phase: number;
  layer: number;
}

// Global speed multiplier: 0.75 (original was 0.5, now 50% faster)
const SPEED_MULT = 0.75;

// Custom Chromatic Aberration Shader
const ChromaticAberrationShader = {
  uniforms: {
    tDiffuse: { value: null },
    uAmount: { value: 0.0022 },
    uTime: { value: 0 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uAmount;
    uniform float uTime;
    varying vec2 vUv;

    void main() {
      vec2 dir = vUv - 0.5;
      float dist = length(dir);
      
      // Subtle time-based pulsing
      float pulse = 1.0 + sin(uTime * 0.12) * 0.12;
      float aberration = uAmount * dist * pulse;
      
      float r = texture2D(tDiffuse, vUv - dir * aberration).r;
      float g = texture2D(tDiffuse, vUv).g;
      float b = texture2D(tDiffuse, vUv + dir * aberration).b;
      float a = texture2D(tDiffuse, vUv).a;
      
      gl_FragColor = vec4(r, g, b, a);
    }
  `,
};

// Film Grain Shader
const FilmGrainShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uIntensity: { value: 0.032 },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform float uIntensity;
    varying vec2 vUv;

    float random(vec2 co) {
      return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      float grain = random(vUv + fract(uTime * 0.5)) * 2.0 - 1.0;
      color.rgb += grain * uIntensity;
      gl_FragColor = color;
    }
  `,
};

export function HeroScene({ className }: HeroSceneProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const scene = new THREE.Scene();

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    camera.position.set(0, 0, 1);

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.setClearColor(0x000000, 0);

    // Enhanced post-processing chain
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.55, 0.75, 0.32);
    composer.addPass(bloomPass);

    // Chromatic Aberration Pass
    const chromaticPass = new ShaderPass(ChromaticAberrationShader);
    composer.addPass(chromaticPass);

    // Film Grain Pass
    const filmGrainPass = new ShaderPass(FilmGrainShader);
    composer.addPass(filmGrainPass);

    // Noise texture for flow field
    const noiseSize = 128;
    const noiseData = new Uint8Array(noiseSize * noiseSize * 4);
    for (let i = 0; i < noiseSize * noiseSize; i += 1) {
      const value = Math.floor(Math.random() * 255);
      const idx = i * 4;
      noiseData[idx] = value;
      noiseData[idx + 1] = value;
      noiseData[idx + 2] = value;
      noiseData[idx + 3] = 255;
    }

    const noiseTexture = new THREE.DataTexture(noiseData, noiseSize, noiseSize, THREE.RGBAFormat);
    noiseTexture.wrapS = THREE.RepeatWrapping;
    noiseTexture.wrapT = THREE.RepeatWrapping;
    noiseTexture.magFilter = THREE.LinearFilter;
    noiseTexture.minFilter = THREE.LinearMipMapLinearFilter;
    noiseTexture.generateMipmaps = true;
    noiseTexture.needsUpdate = true;

    // Create particle textures for different layers
    const createParticleTexture = (innerColor: string, outerColor: string, size: number = 64) => {
      const particleCanvas = document.createElement("canvas");
      particleCanvas.width = size;
      particleCanvas.height = size;
      const ctx = particleCanvas.getContext("2d");

      if (ctx) {
        const half = size / 2;
        const gradient = ctx.createRadialGradient(half, half, 2, half, half, half - 2);
        gradient.addColorStop(0, innerColor);
        gradient.addColorStop(0.5, outerColor);
        gradient.addColorStop(1, "rgba(152, 132, 229, 0)");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, size, size);
      }

      const texture = new THREE.CanvasTexture(particleCanvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.magFilter = THREE.LinearFilter;
      texture.minFilter = THREE.LinearMipMapLinearFilter;
      return texture;
    };

    const particleTextureBg = createParticleTexture("rgba(122, 109, 216, 1)", "rgba(100, 88, 180, 0.7)");
    const particleTextureMid = createParticleTexture("rgba(247, 242, 255, 1)", "rgba(152, 132, 229, 0.9)");
    const particleTextureFg = createParticleTexture("rgba(240, 232, 255, 1)", "rgba(200, 180, 255, 0.85)");

    const planeGeometry = new THREE.PlaneGeometry(2, 2);

    // Enhanced flow material with all effects including Aurora and Metaballs
    const flowMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uPointer: { value: new THREE.Vector2(0, 0) },
        uBoost: { value: 0 },
        uNoise: { value: noiseTexture },
        uBase: { value: new THREE.Color("#080d18") },
        uDeep: { value: new THREE.Color("#0c1422") },
        uAccent: { value: new THREE.Color("#9884e5") },
        uOrb1Pos: { value: new THREE.Vector2(-0.3, 0.4) },
        uOrb2Pos: { value: new THREE.Vector2(0.5, -0.2) },
        uOrb3Pos: { value: new THREE.Vector2(0.1, 0.6) },
        uOrb4Pos: { value: new THREE.Vector2(-0.6, -0.4) },
        // Metaball positions
        uMeta1: { value: new THREE.Vector2(0.0, 0.0) },
        uMeta2: { value: new THREE.Vector2(0.3, 0.2) },
        uMeta3: { value: new THREE.Vector2(-0.2, -0.3) },
        uMeta4: { value: new THREE.Vector2(0.4, -0.1) },
        uMeta5: { value: new THREE.Vector2(-0.35, 0.25) },
      },
      transparent: true,
      depthWrite: false,
      vertexShader: `
        varying vec2 vUv;

        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        precision highp float;

        uniform float uTime;
        uniform vec2 uResolution;
        uniform vec2 uPointer;
        uniform float uBoost;
        uniform sampler2D uNoise;
        uniform vec3 uBase;
        uniform vec3 uDeep;
        uniform vec3 uAccent;
        uniform vec2 uOrb1Pos;
        uniform vec2 uOrb2Pos;
        uniform vec2 uOrb3Pos;
        uniform vec2 uOrb4Pos;
        uniform vec2 uMeta1;
        uniform vec2 uMeta2;
        uniform vec2 uMeta3;
        uniform vec2 uMeta4;
        uniform vec2 uMeta5;
        varying vec2 vUv;

        // Speed multiplier
        const float SPEED = ${SPEED_MULT.toFixed(2)};

        // Hash function for noise
        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        // Value noise
        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          
          float a = hash(i);
          float b = hash(i + vec2(1.0, 0.0));
          float c = hash(i + vec2(0.0, 1.0));
          float d = hash(i + vec2(1.0, 1.0));
          
          return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        // Fractal Brownian Motion (fbm)
        float fbm(vec2 p) {
          float value = 0.0;
          float amplitude = 0.5;
          float frequency = 1.0;
          
          for (int i = 0; i < 6; i++) {
            value += amplitude * noise(p * frequency);
            frequency *= 2.0;
            amplitude *= 0.5;
          }
          return value;
        }

        // Floating gradient orb effect
        float orb(vec2 uv, vec2 center, float radius, float softness) {
          float dist = length(uv - center);
          return smoothstep(radius, radius * softness, dist);
        }

        // Iridescence effect based on position and angle
        vec3 iridescence(vec2 uv, float time) {
          float angle = atan(uv.y, uv.x);
          float dist = length(uv);
          
          float r = sin(angle * 2.0 + dist * 3.0 + time * 0.12 * SPEED) * 0.5 + 0.5;
          float g = sin(angle * 2.0 + dist * 3.0 + time * 0.12 * SPEED + 2.094) * 0.5 + 0.5;
          float b = sin(angle * 2.0 + dist * 3.0 + time * 0.12 * SPEED + 4.188) * 0.5 + 0.5;
          
          return vec3(r, g, b) * 0.07;
        }

        // C-LEVEL: Aurora Borealis Effect
        vec3 aurora(vec2 uv, float time) {
          float t = time * SPEED;
          vec3 auroraColor = vec3(0.0);
          
          // Multiple layered curtains
          for (int i = 0; i < 4; i++) {
            float fi = float(i);
            float offset = fi * 0.8;
            
            // Vertical wave pattern
            float wave = sin(uv.x * 3.0 + t * 0.15 + offset) * 0.5 + 0.5;
            wave *= sin(uv.x * 7.0 - t * 0.08 + offset * 2.0) * 0.5 + 0.5;
            wave *= sin(uv.x * 2.0 + t * 0.2 + offset * 0.5) * 0.5 + 0.5;
            
            // Fade with height - aurora appears in upper portion
            float heightFade = smoothstep(-0.2, 0.8, uv.y + wave * 0.3);
            float topFade = smoothstep(1.2, 0.6, uv.y);
            float curtain = heightFade * topFade;
            
            // Horizontal shimmer
            float shimmer = noise(vec2(uv.x * 10.0 + t * 0.5, uv.y * 3.0 + fi)) * 0.5 + 0.5;
            curtain *= shimmer;
            
            // Aurora color palette - greens, blues, purples
            vec3 color1 = vec3(0.2, 0.8, 0.4); // Green
            vec3 color2 = vec3(0.3, 0.5, 0.9); // Blue
            vec3 color3 = vec3(0.6, 0.3, 0.8); // Purple
            vec3 color4 = vec3(0.8, 0.4, 0.6); // Pink
            
            float colorMix = sin(uv.x * 2.0 + t * 0.1 + fi) * 0.5 + 0.5;
            float colorMix2 = sin(uv.y * 3.0 - t * 0.08) * 0.5 + 0.5;
            
            vec3 layerColor = mix(
              mix(color1, color2, colorMix),
              mix(color3, color4, colorMix),
              colorMix2
            );
            
            auroraColor += layerColor * curtain * (0.15 - fi * 0.025);
          }
          
          return auroraColor;
        }

        // C-LEVEL: Metaball / Liquid Blob Effect (SDF-based)
        float metaball(vec2 uv, vec2 center, float radius) {
          float dist = length(uv - center);
          return radius / (dist * dist + 0.001);
        }

        vec4 metaballs(vec2 uv, float time) {
          float t = time * SPEED;
          
          // Animate metaball positions
          vec2 m1 = uMeta1 + vec2(sin(t * 0.15) * 0.4, cos(t * 0.12) * 0.3);
          vec2 m2 = uMeta2 + vec2(cos(t * 0.18) * 0.35, sin(t * 0.14) * 0.4);
          vec2 m3 = uMeta3 + vec2(sin(t * 0.11 + 1.0) * 0.45, cos(t * 0.16 + 0.5) * 0.35);
          vec2 m4 = uMeta4 + vec2(cos(t * 0.13 + 2.0) * 0.3, sin(t * 0.19 + 1.5) * 0.38);
          vec2 m5 = uMeta5 + vec2(sin(t * 0.17 + 0.8) * 0.38, cos(t * 0.1 + 2.0) * 0.32);
          
          // Calculate metaball field
          float field = 0.0;
          field += metaball(uv, m1, 0.035);
          field += metaball(uv, m2, 0.028);
          field += metaball(uv, m3, 0.032);
          field += metaball(uv, m4, 0.025);
          field += metaball(uv, m5, 0.03);
          
          // Pointer interaction - add dynamic metaball near cursor
          vec2 pointerMeta = uPointer + vec2(sin(t * 0.3) * 0.05, cos(t * 0.25) * 0.05);
          field += metaball(uv, pointerMeta, 0.02 + uBoost * 0.015);
          
          // Create edge glow effect
          float threshold = 0.8;
          float edge = smoothstep(threshold - 0.3, threshold, field) - smoothstep(threshold, threshold + 0.4, field);
          float core = smoothstep(threshold, threshold + 0.6, field);
          
          // Metaball colors - ethereal plasma look
          vec3 edgeColor = vec3(0.5, 0.35, 0.85);
          vec3 coreColor = vec3(0.7, 0.55, 0.95);
          
          // Add color variation based on position
          float colorVar = sin(uv.x * 3.0 + uv.y * 2.0 + t * 0.2) * 0.5 + 0.5;
          edgeColor = mix(edgeColor, vec3(0.4, 0.6, 0.9), colorVar * 0.4);
          coreColor = mix(coreColor, vec3(0.85, 0.7, 1.0), colorVar * 0.3);
          
          vec3 metaColor = edgeColor * edge * 0.6 + coreColor * core * 0.4;
          float metaAlpha = edge * 0.7 + core * 0.5;
          
          return vec4(metaColor, metaAlpha);
        }

        // Enhanced strand field with fbm - uniform distribution
        float strandField(vec2 p, float t, float n) {
          vec2 q = p;
          float value = 0.0;
          
          float slowT = t * SPEED;
          
          // Normalize coordinates to reduce position-dependent density
          vec2 normalizedP = p * 0.85;

          for (int i = 0; i < 7; i++) {
            float fi = float(i);
            float fbmVal = fbm(q * 0.4 + slowT * 0.025);
            
            // Use more balanced frequency multipliers
            q += 0.20 * vec2(
              sin((normalizedP.y * (1.8 + fi * 0.08)) + slowT * 0.35 + fi * 1.1 + n * 1.5 + fbmVal),
              cos((normalizedP.x * (2.0 + fi * 0.06)) - slowT * 0.32 - fi * 1.2 - n * 1.4 - fbmVal)
            );
            
            // Lower frequency for more uniform line distribution
            float line = abs(sin((q.x * 5.5) + (q.y * 3.2) + fi * 0.7));
            
            // Wider smoothstep range for better visibility
            value += smoothstep(0.985, 1.0, line);
          }

          return value / 7.0;
        }

        void main() {
          vec2 uv = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;
          float aspect = uResolution.x / max(uResolution.y, 1.0);
          uv.x *= aspect;
          
          float slowTime = uTime * SPEED;

          vec2 pointer = vec2(uPointer.x * aspect, uPointer.y);
          vec2 delta = pointer - uv;
          float pDist = length(delta) + 0.0001;

          float pull = exp(-pDist * 2.2) * (0.36 + uBoost * 0.85);
          vec2 p = uv + normalize(delta) * pull * 0.48;

          // Use fbm noise with position compensation
          vec2 noiseUv = (p * 0.15) + vec2(slowTime * 0.008, -slowTime * 0.01);
          float n = fbm(noiseUv * 6.0);

          p += vec2(cos(slowTime * 0.16 + n), sin(slowTime * 0.14 - n)) * (0.05 + uBoost * 0.07);

          // Position-based density compensation: boost strands in lower-left, reduce in upper-right
          float positionCompensation = 1.0 + (uv.x + uv.y) * -0.12;
          
          float strands = strandField(p * 0.95, uTime, n) * positionCompensation;
          float core = smoothstep(0.54, 0.02, pDist) * (0.2 + uBoost * 0.7);
          float glow = strands * (0.52 + uBoost * 0.78) + core;

          // Base color gradient
          vec3 color = mix(uBase, uDeep, 0.52 + uv.y * 0.14);
          
          // Floating gradient orbs
          vec2 orb1Animated = uOrb1Pos + vec2(sin(slowTime * 0.12) * 0.15, cos(slowTime * 0.09) * 0.12);
          vec2 orb2Animated = uOrb2Pos + vec2(cos(slowTime * 0.105) * 0.18, sin(slowTime * 0.135) * 0.14);
          vec2 orb3Animated = uOrb3Pos + vec2(sin(slowTime * 0.075) * 0.12, cos(slowTime * 0.12) * 0.16);
          vec2 orb4Animated = uOrb4Pos + vec2(cos(slowTime * 0.09) * 0.14, sin(slowTime * 0.105) * 0.1);
          
          float orb1 = orb(uv, orb1Animated, 0.8, 0.1);
          float orb2 = orb(uv, orb2Animated, 0.65, 0.08);
          float orb3 = orb(uv, orb3Animated, 0.55, 0.12);
          float orb4 = orb(uv, orb4Animated, 0.7, 0.06);
          
          vec3 orbColor1 = vec3(0.42, 0.32, 0.82) * orb1 * 0.16;
          vec3 orbColor2 = vec3(0.55, 0.45, 0.88) * orb2 * 0.14;
          vec3 orbColor3 = vec3(0.35, 0.25, 0.75) * orb3 * 0.11;
          vec3 orbColor4 = vec3(0.48, 0.38, 0.84) * orb4 * 0.13;
          
          color += orbColor1 + orbColor2 + orbColor3 + orbColor4;
          
          // Add iridescence
          vec3 iridColor = iridescence(uv, uTime);
          color += iridColor * (0.55 + glow * 0.35);
          
          // Add aurora effect (C-level)
          vec3 auroraEffect = aurora(uv, uTime);
          color += auroraEffect * 0.65;
          
          // Add metaballs effect (C-level)
          vec4 metaEffect = metaballs(uv, uTime);
          color = mix(color, color + metaEffect.rgb, metaEffect.a * 0.7);
          
          // Add accent glow
          color += uAccent * glow * 1.15;

          // Softer vignette for more uniform appearance
          float vignette = smoothstep(2.2, 0.35, length(uv));
          color *= 0.52 + vignette * 0.48;

          gl_FragColor = vec4(color, 0.97);
        }
      `,
    });

    const plane = new THREE.Mesh(planeGeometry, flowMaterial);
    scene.add(plane);

    // Multi-layer parallax particle system
    const layerConfigs = [
      { count: 400, size: 0.014, opacity: 0.45, speedMult: 0.3 * SPEED_MULT, color: "#8a7ad8", texture: particleTextureBg },
      { count: 600, size: 0.020, opacity: 0.78, speedMult: 0.5 * SPEED_MULT, color: "#c8bbff", texture: particleTextureMid },
      { count: 180, size: 0.032, opacity: 0.88, speedMult: 0.7 * SPEED_MULT, color: "#f0e8ff", texture: particleTextureFg },
    ];

    const particleSystems: {
      points: THREE.Points;
      seeds: FlowParticle[];
      geometry: THREE.BufferGeometry;
      material: THREE.PointsMaterial;
      speedMult: number;
    }[] = [];

    layerConfigs.forEach((config, layerIndex) => {
      const positions = new Float32Array(config.count * 3);
      const seeds: FlowParticle[] = Array.from({ length: config.count }, () => ({
        x: 0,
        y: 0,
        speed: (0.0028 + Math.random() * 0.0044) * config.speedMult,
        phase: Math.random() * Math.PI * 2,
        layer: layerIndex,
      }));

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

      const material = new THREE.PointsMaterial({
        map: config.texture,
        color: config.color,
        transparent: true,
        opacity: config.opacity,
        size: config.size,
        sizeAttenuation: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });

      const points = new THREE.Points(geometry, material);
      scene.add(points);

      particleSystems.push({
        points,
        seeds,
        geometry,
        material,
        speedMult: config.speedMult,
      });
    });

    let aspect = 1;
    const pointerTarget = new THREE.Vector2(0, 0);
    const pointerCurrent = new THREE.Vector2(0, 0);
    let pointerBoost = 0;

    const seedParticle = (particle: FlowParticle) => {
      particle.x = (Math.random() * 2 - 1) * aspect;
      particle.y = Math.random() * 2 - 1;
      particle.speed = (0.0028 + Math.random() * 0.0044) * layerConfigs[particle.layer].speedMult;
      particle.phase = Math.random() * Math.PI * 2;
    };

    const seedAllParticles = () => {
      particleSystems.forEach((system) => {
        const positions = system.geometry.attributes.position as THREE.BufferAttribute;
        system.seeds.forEach((particle, index) => {
          seedParticle(particle);
          positions.setXYZ(index, particle.x, particle.y, 0);
        });
        positions.needsUpdate = true;
      });
    };

    const sampleField = (x: number, y: number, t: number, layerOffset: number) => {
      const slowT = t * SPEED_MULT;
      const base = Math.sin((y * 2.45) + slowT * 0.465 + layerOffset) + Math.cos((x * 2.85) - slowT * 0.41 - layerOffset);

      const dx = x - pointerCurrent.x * aspect;
      const dy = y - pointerCurrent.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.0001;

      const influence = Math.exp(-dist * 2.35) * (0.95 + pointerBoost * 1.4);
      const swirl = Math.atan2(dy, dx) + Math.PI / 2;

      return base + swirl * influence;
    };

    const onPointerMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        return;
      }

      pointerTarget.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointerTarget.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
      pointerBoost = Math.min(pointerBoost + 0.38, 2.4);
    };

    const onPointerLeave = () => {
      pointerTarget.set(0, 0);
      pointerBoost = Math.max(pointerBoost, 0.7);
    };

    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerleave", onPointerLeave);

    const setSize = () => {
      const width = canvas.clientWidth || window.innerWidth;
      const height = canvas.clientHeight || window.innerHeight;
      const pixelRatio = Math.min(window.devicePixelRatio, 1.8);

      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(width, height, false);

      composer.setSize(width, height);
      bloomPass.setSize(width, height);

      aspect = width / Math.max(height, 1);
      camera.left = -aspect;
      camera.right = aspect;
      camera.top = 1;
      camera.bottom = -1;
      camera.updateProjectionMatrix();

      flowMaterial.uniforms.uResolution.value.set(width * pixelRatio, height * pixelRatio);

      seedAllParticles();
    };

    setSize();
    window.addEventListener("resize", setSize);

    const clock = new THREE.Clock();
    let frameId = 0;

    const animate = () => {
      const elapsed = clock.getElapsedTime();
      const delta = Math.min(clock.getDelta(), 0.05);

      pointerCurrent.lerp(pointerTarget, 1 - Math.exp(-delta * 10));
      pointerBoost = Math.max(0, pointerBoost - delta * 0.85);

      flowMaterial.uniforms.uTime.value = elapsed;
      flowMaterial.uniforms.uPointer.value.copy(pointerCurrent);
      flowMaterial.uniforms.uBoost.value = pointerBoost;

      // Update post-processing uniforms
      chromaticPass.uniforms.uTime.value = elapsed;
      filmGrainPass.uniforms.uTime.value = elapsed;

      // Update all particle layers
      particleSystems.forEach((system, layerIndex) => {
        const position = system.geometry.attributes.position as THREE.BufferAttribute;
        const layerOffset = layerIndex * 0.8;

        system.seeds.forEach((particle, index) => {
          const angle = sampleField(particle.x, particle.y, elapsed + particle.phase, layerOffset);
          const velocity = particle.speed * (1 + pointerBoost * 0.5);

          particle.x += Math.cos(angle) * velocity * delta * 60;
          particle.y += Math.sin(angle) * velocity * delta * 60;

          if (Math.abs(particle.x) > aspect + 0.35 || Math.abs(particle.y) > 1.35) {
            seedParticle(particle);
          }

          position.setXYZ(index, particle.x, particle.y, 0);
        });

        position.needsUpdate = true;
      });

      composer.render();
      frameId = window.requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.cancelAnimationFrame(frameId);
      window.removeEventListener("resize", setSize);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerleave", onPointerLeave);

      composer.dispose();
      renderer.dispose();

      // Dispose all resources
      planeGeometry.dispose();
      flowMaterial.dispose();
      noiseTexture.dispose();
      particleTextureBg.dispose();
      particleTextureMid.dispose();
      particleTextureFg.dispose();

      particleSystems.forEach((system) => {
        system.geometry.dispose();
        system.material.dispose();
      });
    };
  }, []);

  return <canvas ref={canvasRef} aria-hidden="true" className={cn("h-full w-full pointer-events-none", className)} />;
}
