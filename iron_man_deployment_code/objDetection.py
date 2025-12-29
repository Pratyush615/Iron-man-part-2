import socket
import threading
import numpy as np
import torch
import clip
from PIL import Image
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import struct
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_name = "ViT-B/32"
model, preprocess = clip.load(model_name, device=device)

model_yolo = YOLO("yolov12n.pt")
model_yolo.to(device)

labels = []
with open('obj_detect_csv/dict.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    labels = [row[0].strip() for row in reader if row and row[0].strip()]

with torch.inference_mode():
    text_tokens = clip.tokenize(labels).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)


class FrameServer:
    def __init__(self, host='10.0.0.61', port=8888):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #Increase buffer size for large frames
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)  # 2MB buffer
        #Set socket timeout to handle disconnections
        self.socket.settimeout(1.0)
        self.running = True
        self.frame_buffer = {}  #store partial frames      

    def detect_objects_in_frame(self, frame):
        try:
            yolo_preds = model_yolo(frame, verbose=False)[0]
            
            if not yolo_preds.boxes or len(yolo_preds.boxes) == 0:
                return []

            crops_pil = []
            boxes_coords = []

            for i, box in enumerate(yolo_preds.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(x1 + 1, min(x2, frame.shape[1]))
                y2 = max(y1 + 1, min(y2, frame.shape[0]))
                
                crop_cv = frame[y1:y2, x1:x2]
                if crop_cv.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                crops_pil.append(crop_pil)
                boxes_coords.append((x1, y1, x2, y2))

            if not crops_pil:
                return []

            #process crops with CLIP
            image_inputs = torch.stack([preprocess(crop).to(device) for crop in crops_pil])

            with torch.inference_mode():
                image_features = model.encode_image(image_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                probs = logits.softmax(dim=-1)
                top_probs, top_labels_indices = probs.topk(1, dim=-1)

            detections = []
            for i, (x1, y1, x2, y2) in enumerate(boxes_coords):
                if i < len(top_labels_indices):
                    label_index = top_labels_indices[i].item()
                    label_text = labels[label_index]
                    confidence = top_probs[i].item()
                    detections.append((x1, y1, x2, y2, label_text, confidence))
                    print(f"🏷️  Detection {i+1}: {label_text} ({confidence:.3f}) at [{x1},{y1},{x2},{y2}]")
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def process_frame_data(self, frame_data):
        """Process received frame data and return detection results"""
        try:
            #Decode JPEG frame
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return "decode_error"
            
            #run object detection
            detections = self.detect_objects_in_frame(frame)
            
            if not detections:
                return "no_detections"
            
            result_parts = []
            for x1, y1, x2, y2, label, confidence in detections:
                result_parts.append(f"{x1},{y1},{x2},{y2},{label},{confidence:.3f}")
            
            result = ";".join(result_parts)
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return "processing_error"

    def handle_udp_packet(self, data, addr):
        """Handle incoming UDP packet - reconstruct frame from chunks"""
        try:
            addr_key = f"{addr[0]}:{addr[1]}"
            
            
            #Check if this is a size header (4 bytes)
            if len(data) == 4:
                frame_size = struct.unpack('<I', data)[0] #Little-endian (i love this name for some reason)
                32
                #Initialize frame buffer
                self.frame_buffer[addr_key] = {
                    'expected_size': frame_size,
                    'received_data': bytearray(),
                    'last_update': time.time()
                }
                return
            
            #Check if we have a frame buffer
            if addr_key not in self.frame_buffer:
                print(f"received data chunk without size header from {addr_key} (size: {len(data)})")
                if len(data) > 4:
                    try:
                        #Check if first 4 bytes look like a reasonable frame size
                        potential_size = struct.unpack('<I', data[:4])[0]
                        if 1000 <= potential_size <= 1000000:  #JPEG size range, tweak this checking later
                            print(f"Extracted frame size from combined packet: {potential_size}")
                            self.frame_buffer[addr_key] = {
                                'expected_size': potential_size,
                                'received_data': bytearray(data[4:]),
                                'last_update': time.time()
                            }
                        else:
                            print(f"wrong frame size: {potential_size}")
                            return
                    except Exception as e:
                        print(f"couldnt extract: {e}")
                        return
                else:
                    print("data too small")
                    return
            else:
                #Append frame data chunk
                buffer_info = self.frame_buffer[addr_key]
                buffer_info['received_data'].extend(data)
                buffer_info['last_update'] = time.time()
            
            #Check if we have complete frame
            buffer_info = self.frame_buffer[addr_key]
            if len(buffer_info['received_data']) >= buffer_info['expected_size']:
                frame_data = bytes(buffer_info['received_data'][:buffer_info['expected_size']])
                                
                #Process the complete frame
                result = self.process_frame_data(frame_data)
                
                #send response back to helmet
                try:
                    response = result.encode('utf-8')
                    bytes_sent = self.socket.sendto(response, addr)
                except Exception as e:
                    print(f"error sending response: {e}")
                
                # Clean up buffer
                del self.frame_buffer[addr_key]
                
        except Exception as e:
            print(f"UDP packet handling error: {e}")
            import traceback
            traceback.print_exc()
            # Clean up buffer on error
            addr_key = f"{addr[0]}:{addr[1]}"
            if addr_key in self.frame_buffer:
                del self.frame_buffer[addr_key]

    def cleanup_old_buffers(self):
        """Remove old incomplete frame buffers"""
        current_time = time.time()
        timeout = 5.0  
        
        to_remove = []
        for addr_key, buffer_info in self.frame_buffer.items():
            if current_time - buffer_info['last_update'] > timeout:
                to_remove.append(addr_key)
        
        for addr_key in to_remove:
            del self.frame_buffer[addr_key]

    def start(self):
        """Start the UDP server"""
        try:
            self.socket.bind((self.host, self.port))
            print(f"listening on {self.host}:{self.port}")
            last_cleanup = time.time()
            
            while self.running:
                try:
                    #receive UDP packet
                    data, addr = self.socket.recvfrom(65536)  #Max UDP packet size
                    
                    self.handle_udp_packet(data, addr)
                    
                    #cleanup of old buffers
                    current_time = time.time()
                    if current_time - last_cleanup > 10.0:
                        self.cleanup_old_buffers()
                        last_cleanup = current_time
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Server error: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except KeyboardInterrupt:
            print("shutting down server...")
        except Exception as e:
            print(f"Server startup error: {e}")
        finally:
            self.running = False
            self.socket.close()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    print("starting")
    
    server = FrameServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("shutting down..")
    finally:
        server.stop()