import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { DangerInterval, Frame, Project } from "./types";

type RobotSceneProps = {
  project: Project;
  frame: Frame | null;
  invalidTime: number | null;
  dangerIntervals: DangerInterval[];
};

const materialCache = new Map<string, THREE.MeshBasicMaterial>();

function material(color: number) {
  const key = String(color);
  if (!materialCache.has(key)) {
    materialCache.set(key, new THREE.MeshBasicMaterial({ color }));
  }
  return materialCache.get(key)!;
}

function createLink(a: [number, number], b: [number, number], radius: number, color: number) {
  const start = new THREE.Vector3(a[0], a[1], 0);
  const end = new THREE.Vector3(b[0], b[1], 0);
  const direction = new THREE.Vector3().subVectors(end, start);
  const length = direction.length();
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 18);
  const mesh = new THREE.Mesh(geometry, material(color));
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  return mesh;
}

export default function RobotScene({ project, frame, invalidTime, dangerIntervals }: RobotSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const worldRef = useRef<THREE.Group | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;
    const world = new THREE.Group();
    scene.add(world);
    worldRef.current = world;
    scene.background = new THREE.Color(0x10151d);
    const camera = new THREE.OrthographicCamera(-3.5, 3.5, 2.5, -2.5, 0.1, 20);
    camera.position.z = 6;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    mount.appendChild(renderer.domElement);

    const grid = new THREE.GridHelper(7, 28, 0x354052, 0x222a36);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const resize = () => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      renderer.setSize(width, height, false);
      const aspect = width / Math.max(height, 1);
      camera.left = -3.5 * aspect;
      camera.right = 3.5 * aspect;
      camera.top = 2.5;
      camera.bottom = -2.5;
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener("resize", resize);
    renderer.setAnimationLoop(() => renderer.render(scene, camera));

    return () => {
      sceneRef.current = null;
      worldRef.current = null;
      window.removeEventListener("resize", resize);
      renderer.setAnimationLoop(null);
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    world.clear();
    const currentDanger = frame !== null
      ? dangerIntervals.find(
          (interval) => frame.time >= interval.start_time - 1e-9 && frame.time <= interval.end_time + 1e-9,
        )
      : undefined;
    const afterCollision = invalidTime !== null && frame !== null && frame.time > invalidTime;
    const armColor = afterCollision ? 0x6b7280 : currentDanger ? 0xef4444 : 0x38bdf8;
    const colliding = Boolean(currentDanger);

    if (frame) {
      frame.links.forEach((point, index) => {
        const joint = new THREE.Mesh(new THREE.CircleGeometry(index === 0 ? 0.14 : 0.1, 24), material(0xe2e8f0));
        joint.position.set(point[0], point[1], 0.02);
        world.add(joint);
        if (index > 0) {
          world.add(createLink(frame.links[index - 1], point, project.link_radius, armColor));
        }
      });

      const end = frame.links[frame.links.length - 1];
      const gripper = new THREE.Mesh(
        new THREE.CircleGeometry(project.gripper_radius, 28),
        new THREE.MeshBasicMaterial({ color: frame.carried_object_id ? 0x22c55e : 0xfacc15 }),
      );
      gripper.position.set(end[0], end[1], 0.03);
      world.add(gripper);

      if (frame.object_position && frame.carried_object_id) {
        const carried = project.carried_objects.find((item) => item.id === frame.carried_object_id);
        if (carried) {
          const objectMesh = new THREE.Mesh(
            new THREE.CircleGeometry(carried.radius, 32),
            new THREE.MeshBasicMaterial({ color: colliding ? 0xef4444 : 0x4ade80 }),
          );
          objectMesh.position.set(frame.object_position[0], frame.object_position[1], 0.04);
          world.add(objectMesh);
        }
      }
    }

    project.obstacles.forEach((obstacle) => {
      const mesh = new THREE.Mesh(
        new THREE.CircleGeometry(obstacle.radius, 40),
        new THREE.MeshBasicMaterial({ color: 0xf97316 }),
      );
      mesh.position.set(obstacle.x, obstacle.y, -0.01);
      world.add(mesh);
    });
  }, [frame, invalidTime, project, dangerIntervals]);

  return <div className="scene" ref={mountRef} />;
}
