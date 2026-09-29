import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Obstacle, TrajectorySample } from "./types";

type SceneProps = {
  sample: TrajectorySample | null;
  rejectionSample: TrajectorySample | null;
  obstacles: Obstacle[];
  samples: TrajectorySample[];
};

const v = (point: { x: number; y: number; z?: number }) =>
  new THREE.Vector3(point.x, point.y, point.z ?? 0);

function cylinderBetween(start: THREE.Vector3, end: THREE.Vector3, radius: number, material: THREE.Material) {
  const direction = new THREE.Vector3().subVectors(end, start);
  const geometry = new THREE.CylinderGeometry(radius, radius, direction.length(), 18);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
  return mesh;
}

function poseGroup(sample: TrajectorySample, color: number, opacity: number) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    roughness: 0.45,
    metalness: 0.1,
  });
  for (const link of sample.links) {
    group.add(cylinderBetween(v(link.start), v(link.end), link.radius, material));
  }
  for (const joint of sample.joints) {
    const jointMesh = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(sample.links[0]?.radius ?? 0.06, 0.06) * 1.25, 20, 16),
      new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.35 }),
    );
    jointMesh.position.copy(v(joint));
    group.add(jointMesh);
  }
  return group;
}

export default function Scene({ sample, rejectionSample, obstacles, samples }: SceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    robot: THREE.Group;
    overlay: THREE.Group;
    obstacleGroup: THREE.Group;
    pathLine: THREE.Line | null;
  } | null>(null);

  const obstacleSignature = useMemo(
    () => JSON.stringify(obstacles),
    [obstacles],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf4f7fb);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 8.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
    keyLight.position.set(3, 4, 8);
    scene.add(keyLight);

    const grid = new THREE.GridHelper(8, 16, 0xaab4c0, 0xd7dee8);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.02;
    scene.add(grid);

    const robot = new THREE.Group();
    const overlay = new THREE.Group();
    const obstacleGroup = new THREE.Group();
    scene.add(robot, overlay, obstacleGroup);

    const state = { scene, camera, renderer, robot, overlay, obstacleGroup, pathLine: null as THREE.Line | null };
    stateRef.current = state;

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    let frame = 0;
    const render = () => {
      frame = requestAnimationFrame(render);
      controls.update();
      renderer.render(scene, camera);
    };
    render();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      stateRef.current = null;
    };
  }, []);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    state.robot.clear();
    if (sample) state.robot.add(poseGroup(sample, 0x2563eb, 1));
  }, [sample]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    state.overlay.clear();
    if (rejectionSample) state.overlay.add(poseGroup(rejectionSample, 0xdc2626, 0.72));
  }, [rejectionSample]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    state.obstacleGroup.clear();
    const material = new THREE.MeshStandardMaterial({
      color: 0xf97316,
      transparent: true,
      opacity: 0.8,
      roughness: 0.55,
    });
    for (const obstacle of obstacles) {
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(obstacle.radius, obstacle.radius, 0.08, 40),
        material.clone(),
      );
      mesh.rotation.x = Math.PI / 2;
      mesh.position.set(obstacle.x, obstacle.y, 0);
      state.obstacleGroup.add(mesh);
    }
  }, [obstacleSignature]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    if (state.pathLine) {
      state.scene.remove(state.pathLine);
      state.pathLine.geometry.dispose();
    }
    if (samples.length < 2) return;
    const points = samples.map((item) => v(item.joints[item.joints.length - 1]));
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineDashedMaterial({
      color: 0x16a34a,
      dashSize: 0.07,
      gapSize: 0.05,
      opacity: 0.7,
    });
    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();
    state.pathLine = line;
    state.scene.add(line);
  }, [samples]);

  return <div className="scene-canvas" ref={containerRef} />;
}
