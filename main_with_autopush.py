"""
CrystalBall with Auto-Push Integration

Enhanced main entry point that starts automated git push gateways.
"""
import atexit
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.git_gateway import (
    start_auto_push_daemon,
    stop_auto_push_daemon,
    push_on_milestone,
    AutoPushContext
)
from src.main import main as original_main

def main_with_auto_push():
    """Enhanced main function with automatic git push integration"""
    
    print("🚀 Starting CrystalBall with Auto-Push Integration")
    
    # Start the auto-push daemon
    start_auto_push_daemon()
    
    # Ensure cleanup on exit
    atexit.register(stop_auto_push_daemon)
    
    # Mark startup milestone
    push_on_milestone("CrystalBall analysis session started")
    
    try:
        # Start file watcher if watchdog is available
        try:
            from tools.auto_push_watcher import start_analysis_watcher
            watcher = start_analysis_watcher()
            print("📁 File watcher enabled for automatic result pushing")
        except ImportError:
            print("⚠️  File watcher not available (install 'watchdog' for enhanced auto-push)")
            watcher = None
        
        # Run the main analysis with auto-push context
        with AutoPushContext("CrystalBall analysis session"):
            result = original_main()
            
        # Mark completion milestone
        push_on_milestone("CrystalBall analysis session completed successfully")
        
        return result
        
    except Exception as e:
        # Push error information
        push_on_milestone(f"CrystalBall analysis session failed: {str(e)[:100]}")
        raise
    
    finally:
        # Cleanup
        stop_auto_push_daemon()
        print("✅ Auto-push daemon stopped")

if __name__ == "__main__":
    main_with_auto_push()