from controller import Supervisor
import random
import math


TEAM_DRONE_COUNT = 5
MIN_SPAWN_DIST = 2


class CTFSupervisor:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.spawned_drones = []
        
        self.clean_leftover_drones()
        
    def clean_leftover_drones(self):
        drone_patterns = ["DRONE_ATTACK_", "DRONE_DEFEND_"]
        
        for pattern in drone_patterns:
            for i in range(TEAM_DRONE_COUNT):
                drone = self.supervisor.getFromDef(f"{pattern}{i}")
                if drone:
                    drone.remove()
                    print(f"Cleaned leftover {pattern}{i}")


    def start_episode(self):
        self.cleanup_episode()
        
        root = self.supervisor.getRoot()
        children_field = root.getField("children")
        
        def get_valid_positions(x_range, count):
            positions = []
            attempts = 0
            while len(positions) < count and attempts < 1000:
                x = random.uniform(x_range[0], x_range[1])
                y = random.uniform(-7, 7)  # Within arena bounds (-7.5 to 7.5)
                z = random.uniform(1, 3)  # 1 to 3 height
                
                # Check minimum distance from existing positions
                valid = True
                for pos in positions:
                    dist = math.sqrt((x-pos[0])**2 + (y-pos[1])**2 + (z-pos[2])**2)
                    if dist < MIN_SPAWN_DIST:
                        valid = False
                        break
                
                if valid:
                    positions.append([x, y, z])
                attempts += 1
            
            return positions
        
        # Attack team spawns on left side (x: -12 to -8)
        attack_positions = get_valid_positions([-11.5, -8], TEAM_DRONE_COUNT)
        
        # Defend team spawns on right side (x: 8 to 12)
        defend_positions = get_valid_positions([8, 11.5], TEAM_DRONE_COUNT)
        
        # Spawn attack team (red)
        for i in range(TEAM_DRONE_COUNT):
            pos = attack_positions[i]
            drone_str = f'DEF DRONE_ATTACK_{i} Mavic2Pro {{ translation {pos[0]} {pos[1]} {pos[2]} bodyColor 1 0 0 }}'
            children_field.importMFNodeFromString(-1, drone_str)
            self.spawned_drones.append(self.supervisor.getFromDef(f"DRONE_ATTACK_{i}"))
        
        # Spawn defend team (blue)
        for i in range(TEAM_DRONE_COUNT):
            pos = defend_positions[i]
            drone_str = f'DEF DRONE_DEFEND_{i} Mavic2Pro {{ translation {pos[0]} {pos[1]} {pos[2]} bodyColor 0 0 1 }}'
            children_field.importMFNodeFromString(-1, drone_str)
            self.spawned_drones.append(self.supervisor.getFromDef(f"DRONE_DEFEND_{i}"))
        
        self.supervisor.simulationResetPhysics()
    
    
    def cleanup_episode(self):
        [drone.remove() for drone in self.spawned_drones if drone]
        self.spawned_drones.clear()
    
    def run(self):
        self.start_episode()
        
        while self.supervisor.step(self.timestep) != -1:
            # Main control loop
            pass


if __name__ == "__main__":
    ctf = CTFSupervisor()
    ctf.run()
