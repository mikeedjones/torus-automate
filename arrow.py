import matplotlib.pyplot as plt
# from pyrsistent import T
import hexy as hx
import numpy as np

lookup = np.load('lookup.npy')


axial_cords = np.zeros((7, 7), dtype='int')
for i in range(7):
    axial_cords[i] = range(-3,4)
ax_cords = np.array([axial_cords, axial_cords.T])

ax_cords_map = np.array([axial_cords, axial_cords.T]).T

SE = np.array((0, 1))
SW = np.array((-1, 1))
W = np.array((-1, 0))
NW = np.array((0, -1))
NE = np.array((1, -1))
E = np.array((1, 0))
ALL_DIRECTIONS = np.array([NW, NE, E, SE, SW, W, ])

def make_grid(arr):
    rectangle = np.zeros((len(arr), 7), dtype='int')
    for i in range(len(arr)):
        if i<4:
            rectangle[i:i + 1, 7-len(arr[i]):] = arr[i]
        else:
            rectangle[i:i + 1, :len(arr[i])] = arr[i]
    return rectangle

def get_neighbor(hex, direction):
    return hex + direction

class CyclicInteger:
    """
    A simple helper class for "cycling" an integer through a range of values. Its value will be set to `lower_limit`
    if it increases above `upper_limit`. Its value will be set to `upper_limit` if its value decreases below
    `lower_limit`.
    """
    def __init__(self, initial_value, lower_limit, upper_limit):
        self.value = initial_value
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def increment(self):
        self.value += 1
        if self.value > self.upper_limit:
            self.value = self.lower_limit

    def decrement(self):
        self.value -= 1
        if self.value < self.lower_limit:
            self.value = self.upper_limit


class HexTile(hx.HexTile):
    def __init__(self, axial_coordinates, radius, tile_id):
        super().__init__(axial_coordinates, radius, tile_id)
        self.color='k'

    def set_value(self, value):
        self.value = CyclicInteger(value, 1, 6)

class HexMap:
    def __init__(self, size, values):

        self.hex_map = hx.HexMap()
        self.max_coord = size

        # Get all possible coordinates within `self.max_coord` as radius.
        spiral_coordinates = hx.get_spiral(np.array((0, 0, 0)), 0, self.max_coord)

        # Convert `spiral_coordinates` to axial coordinates, create hexes 
        hexes = []
        axial_coordinates = hx.cube_to_axial(spiral_coordinates)
        for i, axial in enumerate(axial_coordinates):
            hexes.append(HexTile(axial, 1, i))
            
        self.hex_map[np.array(axial_coordinates)] = hexes

        for i in range(7):
            for j in range(7):
                if values[i,j]!=0:
                    self.hex_map[ax_cords_map[i,j]][0].set_value(values[i,j])


    def get_disk(self, center):
        nbs = [get_neighbor(center, dir) for dir in ALL_DIRECTIONS]
        return [center, *nbs]

    def tap(self, ij):
        tapped = self.hex_map[ij][0]
        tap_co = tapped.axial_coordinates
        for coords in self.get_disk(tap_co):
            if len(self.hex_map[coords])>0:
                self.hex_map[coords][0].value.increment()
                if self.hex_map[coords][0].color=='k':
                    self.hex_map[coords][0].color ='r'
    
    def print(self):
        plt.figure(figsize=(10,10))
        theta = np.radians(90)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        proj = 1/np.sqrt(6)*np.array([[np.sqrt(3),0],[1,2]])

        for h in self.hex_map.items():
            hex = h[1]
            coo = hex.axial_coordinates
            coo = np.matmul(proj, coo.T)
            coo = np.matmul(R, coo)
            plt.text(coo[1]/4, coo[0]/4, hex.value.value, c=hex.color, fontsize=40)
        
        plt.axis('off')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.show()



def propagate_1s(hm):

    taps = []

    start = [-2, 2]
    curr = hm.hex_map[np.array([start])][0]
    curr.color = 'b'
    target = get_neighbor(curr.axial_coordinates, SW)
    target = hm.hex_map[target][0]
    target.color='g'

    # hm.print()
    for start in [[-2, 2],[-1, 1],[0, 0],[1, -1],[2, -2],[3,-3]]:
        for dir in [E, NW]:
            curr = hm.hex_map[np.array([start])][0]
            curr.color = 'b'
            target = get_neighbor(curr.axial_coordinates, SW)
            target = hm.hex_map[target][0]
            target.color='g'

            for i in range(7):
                while target.value.value!=1:
                    taps.append(curr.axial_coordinates[0])
                    
                    hm.tap(curr.axial_coordinates)

                for hex in hm.hex_map.items():
                    hex[1].color='k'

                curr = get_neighbor(curr.axial_coordinates, dir)
                if len(hm.hex_map[curr])>0:
                    curr = hm.hex_map[curr][0]
                else:
                    break
                curr.color = 'b'

                target = get_neighbor(curr.axial_coordinates, SW)
                if len(hm.hex_map[target])>0:
                    target = hm.hex_map[target][0]
                else:
                    break
                target.color='g'
    return taps

def transform_taps(taps):
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    proj = 1/np.sqrt(6)*np.array([[np.sqrt(3),0],[1,2]])

    taps = np.array(taps)
    coo = np.matmul(proj, taps.T)
    coo = np.matmul(R, coo)
    coo = coo[[1,0],:]
    coo[1] = -coo[1]

    return coo.T

def solve(rectangle, transform=True):
    hm = HexMap(3, rectangle)
    
    print('map built')

    taps = propagate_1s(hm)
    
    print('first propagation')
    
    bottom = hm.hex_map[np.array([[0,-3],[1,-3],[2,-3],[3,-3]])]
    bottom = [b.value.value for b in bottom]
    
    solution = lookup[np.array([lookup[:,i]==bottom[i] for i in range (4)]).all(axis=0),4:][0]

    for t, loc in zip(solution, [[-3,0],[-3,1],[-3,2],[-3,3]]):
        for n in range(int(t)):
            taps.extend([np.array(loc)])
            hm.tap(np.array(loc))

    print('solved')

    taps.extend(propagate_1s(hm))
    
    print('second propagation')
    
    unique_rows, counts = np.unique(np.array(taps), axis=0, return_counts=True)

    reduced_taps = []
    for t, count in zip(unique_rows, counts%6):
        for i in range(count):
            reduced_taps.append(t)
    taps = reduced_taps
    if transform:
        taps = transform_taps(taps)

    print('optimised')

    return taps
