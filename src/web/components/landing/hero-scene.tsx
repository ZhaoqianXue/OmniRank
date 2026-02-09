"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { cn } from "@/lib/utils";

interface HeroSceneProps {
  className?: string;
}

interface FlowParticle {
  x: number;
  y: number;
  speed: number;
  phase: number;
}

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
    renderer.toneMappingExposure = 1.06;
    renderer.setClearColor(0x000000, 0);

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.62, 0.68, 0.38);
    composer.addPass(bloomPass);

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

    const particleCanvas = document.createElement("canvas");
    particleCanvas.width = 64;
    particleCanvas.height = 64;
    const particleCtx = particleCanvas.getContext("2d");

    if (particleCtx) {
      const gradient = particleCtx.createRadialGradient(32, 32, 3, 32, 32, 30);
      gradient.addColorStop(0, "rgba(247, 242, 255, 1)");
      gradient.addColorStop(0.45, "rgba(152, 132, 229, 0.9)");
      gradient.addColorStop(1, "rgba(152, 132, 229, 0)");
      particleCtx.fillStyle = gradient;
      particleCtx.fillRect(0, 0, 64, 64);
    }

    const particleTexture = new THREE.CanvasTexture(particleCanvas);
    particleTexture.colorSpace = THREE.SRGBColorSpace;
    particleTexture.magFilter = THREE.LinearFilter;
    particleTexture.minFilter = THREE.LinearMipMapLinearFilter;

    const planeGeometry = new THREE.PlaneGeometry(2, 2);
    const flowMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uPointer: { value: new THREE.Vector2(0, 0) },
        uBoost: { value: 0 },
        uNoise: { value: noiseTexture },
        uBase: { value: new THREE.Color("#0b101e") },
        uDeep: { value: new THREE.Color("#10192e") },
        uAccent: { value: new THREE.Color("#9884e5") },
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
        varying vec2 vUv;

        float strandField(vec2 p, float t, float n) {
          vec2 q = p;
          float value = 0.0;

          for (int i = 0; i < 7; i++) {
            float fi = float(i);
            q += 0.24 * vec2(
              sin((q.y * (2.2 + fi * 0.14)) + t * 0.52 + fi * 1.18 + n * 1.8),
              cos((q.x * (2.5 + fi * 0.11)) - t * 0.48 - fi * 1.33 - n * 1.6)
            );
            float line = abs(sin((q.x * 7.8) + (q.y * 4.4) + fi * 0.85));
            value += smoothstep(0.993, 1.0, line);
          }

          return value / 7.0;
        }

        void main() {
          vec2 uv = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;
          float aspect = uResolution.x / max(uResolution.y, 1.0);
          uv.x *= aspect;

          vec2 pointer = vec2(uPointer.x * aspect, uPointer.y);
          vec2 delta = pointer - uv;
          float pDist = length(delta) + 0.0001;

          float pull = exp(-pDist * 2.2) * (0.42 + uBoost * 1.0);
          vec2 p = uv + normalize(delta) * pull * 0.58;

          vec2 noiseUv = (p * 0.22) + vec2(uTime * 0.012, -uTime * 0.015);
          float n = texture2D(uNoise, noiseUv).r;

          p += vec2(cos(uTime * 0.24 + n), sin(uTime * 0.21 - n)) * (0.07 + uBoost * 0.09);

          float strands = strandField(p * 1.15, uTime, n);
          float core = smoothstep(0.54, 0.02, pDist) * (0.25 + uBoost * 0.82);
          float glow = strands * (0.58 + uBoost * 0.95) + core;

          vec3 color = mix(uBase, uDeep, 0.52 + uv.y * 0.16);
          color += uAccent * glow * 1.35;

          float vignette = smoothstep(1.72, 0.24, length(uv));
          color *= 0.46 + vignette * 0.54;

          gl_FragColor = vec4(color, 0.97);
        }
      `,
    });

    const plane = new THREE.Mesh(planeGeometry, flowMaterial);
    scene.add(plane);

    const particleCount = 720;
    const particlePositions = new Float32Array(particleCount * 3);
    const particleSeeds: FlowParticle[] = Array.from({ length: particleCount }, () => ({
      x: 0,
      y: 0,
      speed: 0.0038 + Math.random() * 0.0058,
      phase: Math.random() * Math.PI * 2,
    }));

    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));

    const particleMaterial = new THREE.PointsMaterial({
      map: particleTexture,
      color: "#d8cbff",
      transparent: true,
      opacity: 0.86,
      size: 0.022,
      sizeAttenuation: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const particlePoints = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particlePoints);

    let aspect = 1;
    const pointerTarget = new THREE.Vector2(0, 0);
    const pointerCurrent = new THREE.Vector2(0, 0);
    let pointerBoost = 0;

    const seedParticle = (particle: FlowParticle) => {
      particle.x = (Math.random() * 2 - 1) * aspect;
      particle.y = Math.random() * 2 - 1;
      particle.speed = 0.0038 + Math.random() * 0.0058;
      particle.phase = Math.random() * Math.PI * 2;
    };

    const seedAllParticles = () => {
      particleSeeds.forEach((particle, index) => {
        seedParticle(particle);
        particlePositions[index * 3] = particle.x;
        particlePositions[index * 3 + 1] = particle.y;
        particlePositions[index * 3 + 2] = 0;
      });

      particleGeometry.attributes.position.needsUpdate = true;
    };

    const sampleField = (x: number, y: number, t: number) => {
      const base = Math.sin((y * 2.45) + t * 0.62) + Math.cos((x * 2.85) - t * 0.55);

      const dx = x - pointerCurrent.x * aspect;
      const dy = y - pointerCurrent.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.0001;

      const influence = Math.exp(-dist * 2.35) * (1.15 + pointerBoost * 1.65);
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
      pointerBoost = Math.min(pointerBoost + 0.42, 2.6);
    };

    const onPointerLeave = () => {
      pointerTarget.set(0, 0);
      pointerBoost = Math.max(pointerBoost, 0.8);
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

      pointerCurrent.lerp(pointerTarget, 1 - Math.exp(-delta * 12));
      pointerBoost = Math.max(0, pointerBoost - delta * 0.92);

      flowMaterial.uniforms.uTime.value = elapsed;
      flowMaterial.uniforms.uPointer.value.copy(pointerCurrent);
      flowMaterial.uniforms.uBoost.value = pointerBoost;

      const position = particleGeometry.attributes.position as THREE.BufferAttribute;

      particleSeeds.forEach((particle, index) => {
        const angle = sampleField(particle.x, particle.y, elapsed + particle.phase);
        const velocity = particle.speed * (1 + pointerBoost * 0.6);

        particle.x += Math.cos(angle) * velocity * delta * 60;
        particle.y += Math.sin(angle) * velocity * delta * 60;

        if (Math.abs(particle.x) > aspect + 0.35 || Math.abs(particle.y) > 1.35) {
          seedParticle(particle);
        }

        position.setXYZ(index, particle.x, particle.y, 0);
      });

      position.needsUpdate = true;

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

      [
        planeGeometry,
        flowMaterial,
        particleGeometry,
        particleMaterial,
        noiseTexture,
        particleTexture,
      ].forEach((resource) => resource.dispose());
    };
  }, []);

  return <canvas ref={canvasRef} aria-hidden="true" className={cn("h-full w-full pointer-events-none", className)} />;
}
