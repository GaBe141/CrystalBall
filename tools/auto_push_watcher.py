#!/usr/bin/env python3
"""
File watcher that triggers git pushes when files change during analysis runs.
"""
import os
import sys
import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.automation.git_gateway import push_on_data_change, push_on_results_generated, _gateway

class AnalysisFileHandler(FileSystemEventHandler):
    """Handler for file system events during analysis"""
    
    def __init__(self):
        self.last_push_time = {}
        self.push_cooldown = 30  # 30 seconds between pushes for same file
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        current_time = time.time()
        
        # Rate limiting per file
        if file_path in self.last_push_time:
            if (current_time - self.last_push_time[file_path]) < self.push_cooldown:
                return
        
        self.last_push_time[file_path] = current_time
        
        # Trigger pushes based on file types
        if any(file_name.endswith(ext) for ext in ['.csv', '.json', '.xlsx']):
            if 'processed' in file_path:
                push_on_results_generated()
            elif 'data' in file_path:
                push_on_data_change(file_path)
                
        elif file_name.endswith('.png') or file_name.endswith('.html'):
            if 'visuals' in file_path or 'visualizations' in file_path:
                push_on_results_generated()

class AnalysisWatcher:
    """File system watcher for analysis directories"""
    
    def __init__(self, watch_dirs=None):
        self.watch_dirs = watch_dirs or [
            'data/processed',
            'data/processed/visualizations', 
            'src',
            'config'
        ]
        self.observer = Observer()
        self.handler = AnalysisFileHandler()
        self.running = False
    
    def start(self):
        """Start watching for file changes"""
        if self.running:
            return
            
        base_path = Path.cwd()
        
        for watch_dir in self.watch_dirs:
            watch_path = base_path / watch_dir
            if watch_path.exists():
                self.observer.schedule(self.handler, str(watch_path), recursive=True)
                print(f"📁 Watching: {watch_path}")
        
        self.observer.start()
        self.running = True
        print("🔍 File watcher started - auto-push enabled for analysis outputs")
    
    def stop(self):
        """Stop watching for file changes"""
        if not self.running:
            return
            
        self.observer.stop()
        self.observer.join()
        self.running = False
        print("🛑 File watcher stopped")
    
    def watch_forever(self):
        """Start watching and keep running until interrupted"""
        self.start()
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

def start_analysis_watcher():
    """Start the file watcher for analysis outputs"""
    watcher = AnalysisWatcher()
    
    # Start in background thread
    watch_thread = threading.Thread(target=watcher.watch_forever, daemon=True)
    watch_thread.start()
    
    return watcher

if __name__ == "__main__":
    # Run as standalone script
    watcher = AnalysisWatcher()
    print("Starting CrystalBall file watcher...")
    watcher.watch_forever()