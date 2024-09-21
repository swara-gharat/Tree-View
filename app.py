import http.server
import socketserver
import cgi
from PIL import Image, ImageDraw, ImageOps
from numpy import array, sqrt
import heapq
import io

class OptimalPathing:
    def __init__(self, img):
        self.img = array(img)

    def create_graph(self, binary_image):
        rows, cols = binary_image.shape
        graph = {}
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Only 4 neighbors for simplicity

        for i in range(rows):
            for j in range(cols):
                neighbors = []
                for dx, dy in neighbor_offsets:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        weight = sqrt((ni - rows + 1) ** 2 + (nj - cols + 1) ** 2) + (255 - binary_image[ni, nj])
                        neighbors.append(((ni, nj), weight))
                graph[(i, j)] = neighbors
        return graph

    def trace_path(self, parents, start, target):
        path = []
        current = target
        while current != start:
            path.append(current)
            current = parents[current]
        path.append(start)
        path.reverse()
        return path

    def compute_dijkstra(self, start_pixel=(0, 0), target_pixel=None):
        if target_pixel is None:
            target_pixel = (self.img.shape[0] - 1, self.img.shape[1] - 1)
        graph = self.create_graph(self.img)
        parents = {}
        heap = [(0, start_pixel)]
        visited = set()

        while heap:
            cost, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            if current == target_pixel:
                break

            for neighbor, weight in graph.get(current, []):
                if neighbor not in visited:
                    parents[neighbor] = current
                    heapq.heappush(heap, (cost + weight, neighbor))

        shortest_path = self.trace_path(parents, start_pixel, target_pixel)

        # Visualize the image and the shortest path using PIL
        img = Image.fromarray(self.img)
        img_color = ImageOps.colorize(img.convert('L'), 'black', 'white')

        draw = ImageDraw.Draw(img_color)
        if shortest_path:
            for i in range(len(shortest_path) - 1):
                start_point = (shortest_path[i][1], shortest_path[i][0])
                end_point = (shortest_path[i + 1][1], shortest_path[i + 1][0])
                # Ensure path does not go out of bounds
                start_point = (max(0, min(img_color.width - 1, start_point[0])), max(0, min(img_color.height - 1, start_point[1])))
                end_point = (max(0, min(img_color.width - 1, end_point[0])), max(0, min(img_color.height - 1, end_point[1])))
                draw.line([start_point, end_point], fill="blue", width=2)

        # Save the result to a BytesIO object
        img_bytes = io.BytesIO()
        img_color.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return img_bytes

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_type, pdict = cgi.parse_header(self.headers['Content-Type'])
        if content_type == 'multipart/form-data':
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'})
            file_item = form['file']

            if file_item.file:
                image = Image.open(file_item.file).convert('L')  # Convert to grayscale
                img_array = array(image)

                # Create instance of OptimalPathing
                obj = OptimalPathing(img_array)

                # Compute Dijkstra's path
                result_img = obj.compute_dijkstra()

                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(result_img.getvalue())
            else:
                self.send_error(400, "No file uploaded")
        else:
            self.send_error(400, "Invalid content type")

PORT = 8000
Handler = RequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()
