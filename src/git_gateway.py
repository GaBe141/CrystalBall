"""
Git Push Gateways - Automated backup triggers during code execution
"""
import os
import subprocess
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitGateway:
    """Automated Git push gateway system"""
    
    def __init__(self, repo_path: str = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.last_push = 0
        self.push_interval = 300  # 5 minutes minimum between pushes
        self.pending_changes = False
        self._lock = threading.Lock()
    
    def _has_changes(self) -> bool:
        """Check if there are uncommitted changes"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            return len(result.stdout.strip()) > 0
        except Exception as e:
            logger.warning(f"Failed to check git status: {e}")
            return False
    
    def _commit_and_push(self, message: str = None):
        """Commit changes and push to remote"""
        if not message:
            message = f"Auto-commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            # Add all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=self.repo_path,
                check=True,
                timeout=30
            )
            
            # Commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                check=True,
                timeout=30
            )
            
            # Push
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=self.repo_path,
                check=True,
                timeout=60
            )
            
            logger.info(f"✅ Auto-pushed changes: {message}")
            self.last_push = time.time()
            self.pending_changes = False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git operation failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error during git push: {e}")
    
    def trigger_push(self, message: str = None, force: bool = False):
        """Trigger a git push if conditions are met"""
        with self._lock:
            current_time = time.time()
            
            # Check rate limiting
            if not force and (current_time - self.last_push) < self.push_interval:
                self.pending_changes = True
                return
            
            # Check for changes
            if not self._has_changes():
                return
            
            # Execute push in background thread
            thread = threading.Thread(
                target=self._commit_and_push,
                args=(message,),
                daemon=True
            )
            thread.start()

# Global gateway instance
_gateway = GitGateway()

def auto_push_on_execution(message: str = None):
    """Decorator to trigger git push when function executes"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Trigger push after successful execution
                push_message = message or f"Auto-push after {func.__name__} execution"
                _gateway.trigger_push(push_message)
                return result
            except Exception as e:
                # Still try to push on error (might contain useful debug info)
                error_message = f"Auto-push after {func.__name__} error: {str(e)[:50]}"
                _gateway.trigger_push(error_message)
                raise
        return wrapper
    return decorator

def push_on_milestone(milestone: str):
    """Manually trigger a push at code milestones"""
    message = f"Milestone: {milestone}"
    _gateway.trigger_push(message, force=True)

def push_on_data_change(data_path: str):
    """Trigger push when data files are modified"""
    if os.path.exists(data_path):
        message = f"Data updated: {os.path.basename(data_path)}"
        _gateway.trigger_push(message)

def push_on_results_generated():
    """Trigger push when analysis results are generated"""
    message = "Analysis results generated"
    _gateway.trigger_push(message)

class AutoPushContext:
    """Context manager for automatic pushes during operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"🔄 Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            message = f"Completed: {self.operation_name} ({duration.total_seconds():.1f}s)"
            _gateway.trigger_push(message)
        else:
            message = f"Failed: {self.operation_name} - {str(exc_val)[:50]}"
            _gateway.trigger_push(message)

# Scheduled background pusher
class ScheduledPusher:
    """Background thread that pushes pending changes periodically"""
    
    def __init__(self, interval: int = 600):  # 10 minutes
        self.interval = interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the background pusher"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"📅 Started scheduled pusher (interval: {self.interval}s)")
    
    def stop(self):
        """Stop the background pusher"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _run(self):
        """Background thread loop"""
        while self.running:
            try:
                if _gateway.pending_changes:
                    _gateway.trigger_push("Scheduled auto-push", force=True)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Scheduled pusher error: {e}")
                time.sleep(60)  # Wait before retrying

# Global scheduled pusher
_scheduled_pusher = ScheduledPusher()

def start_auto_push_daemon():
    """Start the background auto-push daemon"""
    _scheduled_pusher.start()

def stop_auto_push_daemon():
    """Stop the background auto-push daemon"""
    _scheduled_pusher.stop()