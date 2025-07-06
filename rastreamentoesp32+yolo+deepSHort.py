import cv2
import time
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import numpy as np
from queue import Queue
import logging

# Configurar logging para debugar
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceTrackingApp:
    """
    Sistema de rastreamento de pessoas com reidentificação facial otimizado
    Separação clara entre lógica de negócio e interface
    """
    
    def __init__(self):
        # ===== Configurações =====
        self.ESP32_STREAM_URL = "http://192.168.0.100:81/stream"
        self.YOLO_MODEL_PATH = "yolov8n.pt"
        self.FACES_PATH = "faces"
        
        # ===== Estado da aplicação =====
        self.tracking = False
        self.paused = False
        self.frame_queue = Queue(maxsize=2)  # Buffer pequeno para evitar delay
        self.stats = {"fps": 0, "pessoas": 0, "faces_id": 0}
        
        # ===== Inicialização dos modelos =====
        try:
            self.model = YOLO(self.YOLO_MODEL_PATH)
            self.tracker = DeepSort(max_age=30, max_iou_distance=0.7)
            logger.info("✅ Modelos YOLO e DeepSort carregados")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")
            raise
        
        # ===== Sistema de reconhecimento facial (opcional) =====
        self.face_recognition_enabled = self._init_face_recognition()
        
        # ===== Interface =====
        self._init_ui()
        
    def _init_face_recognition(self):
        """Inicializa reconhecimento facial com tratamento de erro"""
        try:
            import face_recognition
            self.face_recognition = face_recognition
            self.known_face_encodings = []
            self.known_face_names = []
            self._load_known_faces()
            logger.info("✅ Reconhecimento facial habilitado")
            return True
        except ImportError:
            logger.warning("⚠️ face_recognition não instalado. Continuando sem ReID facial")
            messagebox.showwarning(
                "Módulo Ausente", 
                "face_recognition não encontrado.\n"
                "Instale com: pip install face_recognition\n"
                "Continuando apenas com rastreamento de pessoas."
            )
            return False
    
    def _load_known_faces(self):
        """Carrega faces conhecidas da pasta"""
        if not os.path.exists(self.FACES_PATH):
            os.makedirs(self.FACES_PATH)
            logger.info(f"📁 Pasta {self.FACES_PATH} criada")
            return
            
        for filename in os.listdir(self.FACES_PATH):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    filepath = os.path.join(self.FACES_PATH, filename)
                    image = self.face_recognition.load_image_file(filepath)
                    encodings = self.face_recognition.face_encodings(image)
                    
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        logger.info(f"✅ Face carregada: {name}")
                    else:
                        logger.warning(f"⚠️ Nenhuma face detectada em: {filename}")
                        
                except Exception as e:
                    logger.error(f"❌ Erro ao carregar {filename}: {e}")
    
    def _identify_person(self, face_encoding):
        """Identifica pessoa baseada na codificação facial"""
        if not self.known_face_encodings:
            return "Desconhecido", 0.0
            
        distances = self.face_recognition.face_distance(self.known_face_encodings, face_encoding)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        # Threshold mais rigoroso para melhor precisão
        if min_dist < 0.4:
            return self.known_face_names[min_idx], (1 - min_dist)
        return "Desconhecido", (1 - min_dist)
    
    def _process_frame(self, frame):
        """Processa um frame: detecção + rastreamento + identificação"""
        try:
            # 1. Detecção YOLO (pessoas apenas)
            results = self.model(frame, verbose=False)[0]
            detections = []
            
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = result
                if int(cls) == 0 and score > 0.5:  # Pessoa com confiança > 50%
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append((bbox, score, 'person'))
            
            # 2. Rastreamento DeepSort
            tracks = self.tracker.update_tracks(detections, frame=frame)
            active_tracks = 0
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                active_tracks += 1
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                
                # Desenhar caixa de rastreamento
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 3. Reconhecimento facial (apenas a cada 5 frames para performance)
            faces_identified = 0
            if self.face_recognition_enabled and hasattr(self, '_frame_count'):
                if self._frame_count % 5 == 0:  # Processa reconhecimento facial a cada 5 frames
                    try:
                        # Redimensionar frame para acelerar processamento facial
                        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        face_locations = self.face_recognition.face_locations(small_frame, model="hog")
                        face_encodings = self.face_recognition.face_encodings(small_frame, face_locations)
                        
                        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                            # Escalar coordenadas de volta
                            top, right, bottom, left = top*2, right*2, bottom*2, left*2
                            
                            name, confidence = self._identify_person(encoding)
                            faces_identified += 1
                            
                            # Desenhar identificação facial
                            cv2.rectangle(frame, (left, top), (right, bottom), (255, 200, 0), 2)
                            text = f"{name} ({confidence:.2f})"
                            cv2.putText(frame, text, (left, top - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
                    except Exception as e:
                        logger.error(f"Erro no reconhecimento facial: {e}")
            
            # Atualizar estatísticas
            self.stats["pessoas"] = active_tracks
            self.stats["faces_id"] = faces_identified
            
            return frame
            
        except Exception as e:
            logger.error(f"Erro no processamento do frame: {e}")
            return frame
    
    def _video_capture_thread(self):
        """Thread dedicada para captura de vídeo"""
        cap = None
        try:
            cap = cv2.VideoCapture(self.ESP32_STREAM_URL)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror(
                    "Erro de Conexão", 
                    f"Não foi possível conectar à ESP32-CAM.\nURL: {self.ESP32_STREAM_URL}"
                ))
                return
            
            # Configurar buffer da câmera
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            start_time = time.time()
            frame_count = 0
            self._frame_count = 0
            
            logger.info("🎥 Captura de vídeo iniciada")
            
            while self.tracking:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ Falha na captura do frame")
                    continue
                
                # Processar frame
                processed_frame = self._process_frame(frame)
                
                # Calcular FPS
                frame_count += 1
                self._frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                self.stats["fps"] = fps
                
                # Adicionar informações no frame
                self._add_frame_info(processed_frame, fps)
                
                # Enviar frame para interface (não-bloqueante)
                if not self.frame_queue.full():
                    self.frame_queue.put(processed_frame.copy())
                
        except Exception as e:
            logger.error(f"Erro na captura de vídeo: {e}")
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro na captura: {e}"))
        finally:
            if cap:
                cap.release()
            logger.info("🛑 Captura de vídeo finalizada")
    
    def _add_frame_info(self, frame, fps):
        """Adiciona informações visuais no frame"""
        h, w = frame.shape[:2]
        
        # Fundo semi-transparente para as informações
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Informações de texto
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Pessoas: {self.stats['pessoas']}",
            f"Faces ID: {self.stats['faces_id']}",
            f"Status: {'PAUSADO' if self.paused else 'ATIVO'}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + (i * 20)
            cv2.putText(frame, line, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _update_display(self):
        """Atualiza display da interface (thread principal)"""
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Converter para formato Tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
                
                # Atualizar label
                self.video_label.imgtk = img
                self.video_label.configure(image=img)
                
                # Armazenar frame atual para salvar
                self.current_frame = frame
                
                # Atualizar barra de status
                status_text = f"FPS: {self.stats['fps']:.1f} | Pessoas: {self.stats['pessoas']} | Faces: {self.stats['faces_id']}"
                self.status_var.set(status_text)
                
        except Exception as e:
            logger.error(f"Erro na atualização do display: {e}")
        
        # Agendar próxima atualização
        if self.tracking:
            self.root.after(30, self._update_display)  # ~33 FPS max na interface
    
    def start_tracking(self):
        """Inicia o rastreamento"""
        if not self.tracking:
            self.tracking = True
            self.paused = False
            
            # Iniciar thread de captura
            threading.Thread(target=self._video_capture_thread, daemon=True).start()
            
            # Iniciar atualização da interface
            self._update_display()
            
            # Atualizar botões
            self.btn_start.config(state='disabled')
            self.btn_pause.config(state='normal', text="⏸️ Pausar")
            self.btn_stop.config(state='normal')
            
            logger.info("🚀 Rastreamento iniciado")
    
    def pause_tracking(self):
        """Pausa/resume o rastreamento"""
        self.paused = not self.paused
        text = "▶️ Retomar" if self.paused else "⏸️ Pausar"
        self.btn_pause.config(text=text)
        logger.info(f"⏸️ Rastreamento {'pausado' if self.paused else 'retomado'}")
    
    def stop_tracking(self):
        """Para o rastreamento"""
        self.tracking = False
        self.paused = False
        
        # Atualizar botões
        self.btn_start.config(state='normal')
        self.btn_pause.config(state='disabled', text="⏸️ Pausar")
        self.btn_stop.config(state='disabled')
        
        logger.info("🛑 Rastreamento parado")
    
    def save_frame(self):
        """Salva o frame atual"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            timestamp = int(time.time())
            filename = f"frame_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Sucesso", f"Frame salvo como: {filename}")
            logger.info(f"💾 Frame salvo: {filename}")
        else:
            messagebox.showwarning("Aviso", "Nenhum frame disponível para salvar")
    
    def _init_ui(self):
        """Inicializa a interface gráfica"""
        self.root = tk.Tk()
        self.root.title("🛰️ ESP32-CAM Tracker Pro - YOLO + DeepSORT + Face ReID")
        self.root.geometry("800x700")
        
        # Frame principal do vídeo
        video_frame = tk.Frame(self.root, bg='black')
        video_frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        self.video_label = tk.Label(video_frame, bg='black', text="Pressione 'Iniciar' para começar")
        self.video_label.pack()
        
        # Frame dos controles
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # Botões de controle
        self.btn_start = tk.Button(control_frame, text="▶️ Iniciar", width=12, 
                                  command=self.start_tracking, bg='#4CAF50', fg='white')
        self.btn_start.grid(row=0, column=0, padx=5)
        
        self.btn_pause = tk.Button(control_frame, text="⏸️ Pausar", width=12, 
                                  command=self.pause_tracking, state='disabled')
        self.btn_pause.grid(row=0, column=1, padx=5)
        
        self.btn_stop = tk.Button(control_frame, text="⏹️ Parar", width=12, 
                                 command=self.stop_tracking, state='disabled', bg='#f44336', fg='white')
        self.btn_stop.grid(row=0, column=2, padx=5)
        
        self.btn_save = tk.Button(control_frame, text="💾 Salvar Frame", width=15, 
                                 command=self.save_frame, bg='#2196F3', fg='white')
        self.btn_save.grid(row=0, column=3, padx=5)
        
        # Barra de status
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto para iniciar")
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               relief='sunken', anchor='w')
        status_label.pack(fill='x')
        
        # Frame de informações
        info_frame = tk.LabelFrame(self.root, text="Informações do Sistema")
        info_frame.pack(fill='x', padx=10, pady=5)
        
        face_status = "✅ Habilitado" if self.face_recognition_enabled else "❌ Desabilitado (instale face_recognition)"
        info_text = f"""
        • URL ESP32-CAM: {self.ESP32_STREAM_URL}
        • Modelo YOLO: {self.YOLO_MODEL_PATH}
        • Reconhecimento Facial: {face_status}
        • Images Conhecidas: {len(self.known_face_names) if self.face_recognition_enabled else 0}
        """
        
        info_label = tk.Label(info_frame, text=info_text, justify='left')
        info_label.pack(padx=10, pady=5)
        
        # Configurar fechamento
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _on_closing(self):
        """Gerencia fechamento da aplicação"""
        self.stop_tracking()
        self.root.destroy()
        logger.info("🏁 Aplicação encerrada")
    
    def run(self):
        """Executa a aplicação"""
        logger.info("🚀 Iniciando aplicação...")
        self.root.mainloop()

# ===== SCRIPT DE INSTALAÇÃO AUTOMÁTICA =====
def check_and_install_dependencies():
    """Verifica e instala dependências ausentes"""
    required_packages = {
        'ultralytics': 'ultralytics',
        'deep_sort_realtime': 'deep-sort-realtime',
        'PIL': 'Pillow',
        'cv2': 'opencv-python'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("📦 Instalando dependências ausentes...")
        import subprocess
        import sys
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ Dependências instaladas!")

# ===== EXECUÇÃO PRINCIPAL =====
if __name__ == "__main__":
    try:
        # Verificar dependências
        check_and_install_dependencies()
        
        # Executar aplicação
        app = FaceTrackingApp()
        app.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Aplicação interrompida pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro crítico: {e}")
        messagebox.showerror("Erro Crítico", f"Erro inesperado: {e}")
