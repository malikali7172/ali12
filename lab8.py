
import heapq

class Node:
    def __init__(self, name, heuristic):
        self.name = name
        self.heuristic = heuristic
        self.parent = None

    def __lt__(self, other):
        return self.heuristic < other.heuristic

def best_first_search(start_node, goal_node, graph):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, (start_node.heuristic, start_node))
    
    while open_list:
        _, current_node = heapq.heappop(open_list)
        if current_node.name == goal_node.name:
            path = []
            while current_node:
                path.append(current_node.name)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path
        
        closed_list.add(current_node.name)
        for neighbor_name in graph.get(current_node.name, []):
            if neighbor_name in closed_list:
                continue
            neighbor_node = Node(neighbor_name, heuristic_func(neighbor_name, goal_node.name))
            neighbor_node.parent = current_node
            
            heapq.heappush(open_list, (neighbor_node.heuristic, neighbor_node))
    return None

def heuristic_func(current, goal):
    return abs(len(goal) - len(current))

if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': ['G'],
        'E': ['G'],
        'F': ['G'],
        'G': []
    }
    start_node = Node('A', heuristic_func('A', 'G'))
    goal_node = Node('G', 0)  # Heuristic for goal node is 0
    path = best_first_search(start_node, goal_node, graph)
    
    if path:
        print("Path found:", path)
    else:
        print("No path found")

import heapq

class Node:
    def __init__(self, name, g=0, h=0):
        self.name = name
        self.g = g  
        self.h = h  
        self.f = g + h 
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

def a_star_search(start_node, goal_node, graph, heuristic):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, (start_node.f, start_node))
    
    while open_list:
        _, current_node = heapq.heappop(open_list)
        if current_node.name == goal_node.name:
            path = []
            while current_node:
                path.append(current_node.name)
                current_node = current_node.parent
            return path[::-1] 
        
        closed_list.add(current_node.name)
        for neighbor_name, cost in graph.get(current_node.name, []):
            if neighbor_name in closed_list:
                continue
            g = current_node.g + cost
            h = heuristic(neighbor_name, goal_node.name)
            neighbor_node = Node(neighbor_name, g, h)
            neighbor_node.parent = current_node
            open_node = next((node for f_val, node in open_list if node.name == neighbor_name), None)
            if open_node and open_node.f <= neighbor_node.f:
                continue

            heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
    return None

def heuristic_func(current, goal)
    return abs(len(goal) - len(current))

if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('D', 2), ('E', 5)],
        'C': [('F', 1)],
        'D': [('G', 1)],
        'E': [('G', 2)],
        'F': [('G', 5)],
        'G': []
    }
    start_node = Node('A', 0, heuristic_func('A', 'G'))
    goal_node = Node('G', 0, 0)  # Heuristic for goal node is 0

    # Perform A* Search
    path = a_star_search(start_node, goal_node, graph, heuristic_func)
    
    if path:
        print("Path found:", path)
    else:
        print("No path found")


import heapq
import itertools

class PuzzleState:
    def __init__(self, board, empty_tile, g=0, h=0):
        self.board = board
        self.empty_tile = empty_tile
        self.g = g  
        self.h = h  
        self.f = g + h 
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

    def get_moves(self):
        """Generate all possible moves from the current state."""
        moves = []
        x, y = self.empty_tile
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_board = [row[:] for row in self.board]
                new_board[x][y], new_board[nx][ny] = new_board[nx][ny], new_board[x][y]
                moves.append((new_board, (nx, ny)))
        return moves

    def __repr__(self):
        return '\n'.join(' '.join(str(x) if x != 0 else ' ' for x in row) for row in self.board)

def heuristic(board):
    """Calculate the heuristic value for the given board."""
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    distance = 0
    for i in range(3):
        for j in range(3):
            value = board[i][j]
            if value != 0:
                goal_x, goal_y = divmod(value - 1, 3)
                distance += abs(goal_x - i) + abs(goal_y - j)
    return distance

def a_star_search(start_board):
    start_tile = (0, 0)
    for i in range(3):
        for j in range(3):
            if start_board[i][j] == 0:
                start_tile = (i, j)
    
    start_state = PuzzleState(start_board, start_tile, 0, heuristic(start_board))
    open_list = []
    heapq.heappush(open_list, (start_state.f, start_state))
    closed_list = set()
    
    while open_list:
        _, current_state = heapq.heappop(open_list)
        
        if current_state.board == [[1, 2, 3], [4, 5, 6], [7, 8, 0]]:
            path = []
            while current_state:
                path.append(current_state.board)
                current_state = current_state.parent
            return path[::-1]
        
        closed_list.add(tuple(map(tuple, current_state.board)))
        
        for new_board, new_empty_tile in current_state.get_moves():
            if tuple(map(tuple, new_board)) in closed_list:
                continue
            
            g = current_state.g + 1
            h = heuristic(new_board)
           
