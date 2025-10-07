#!/usr/bin/env python3
"""
Git Gateway CLI Tool for CrystalBall

Manage automated git push gateways from command line.
"""
import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.git_gateway import (
    _gateway,
    start_auto_push_daemon,
    stop_auto_push_daemon,
    push_on_milestone,
    _scheduled_pusher
)

def status():
    """Show current gateway status"""
    print("🔍 Git Gateway Status")
    print("=" * 40)
    
    # Check if git repo exists
    if _gateway._has_changes():
        print("📝 Status: Uncommitted changes detected")
    else:
        print("✅ Status: No pending changes")
    
    # Check scheduled pusher
    if _scheduled_pusher.running:
        print("🤖 Daemon: Running")
    else:
        print("⏸️  Daemon: Stopped")
    
    # Show last push time
    if _gateway.last_push > 0:
        last_push_ago = time.time() - _gateway.last_push
        print(f"⏰ Last push: {last_push_ago:.1f} seconds ago")
    else:
        print("⏰ Last push: Never")
    
    print(f"📂 Repository: {_gateway.repo_path}")
    print(f"⏱️  Push interval: {_gateway.push_interval} seconds")

def start_daemon():
    """Start the auto-push daemon"""
    print("🚀 Starting auto-push daemon...")
    start_auto_push_daemon()
    print("✅ Daemon started")

def stop_daemon():
    """Stop the auto-push daemon"""
    print("⏹️  Stopping auto-push daemon...")
    stop_auto_push_daemon()
    print("✅ Daemon stopped")

def force_push(message=None):
    """Force a git push now"""
    print("⚡ Forcing git push...")
    if message:
        _gateway.trigger_push(message, force=True)
        print(f"✅ Push triggered with message: {message}")
    else:
        _gateway.trigger_push("Manual push via CLI", force=True)
        print("✅ Push triggered")

def milestone(description):
    """Mark a milestone and push"""
    print(f"🎯 Marking milestone: {description}")
    push_on_milestone(description)
    print("✅ Milestone marked and pushed")

def watch():
    """Start file watching mode"""
    try:
        from tools.auto_push_watcher import AnalysisWatcher
        print("👁️  Starting file watcher mode...")
        print("Press Ctrl+C to stop")
        
        watcher = AnalysisWatcher()
        watcher.watch_forever()
        
    except ImportError:
        print("❌ File watcher not available. Install 'watchdog' package:")
        print("   pip install watchdog")
    except KeyboardInterrupt:
        print("\n👋 File watcher stopped")

def main():
    parser = argparse.ArgumentParser(
        description="CrystalBall Git Gateway CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Show current status
  %(prog)s start                     # Start auto-push daemon
  %(prog)s stop                      # Stop auto-push daemon
  %(prog)s push                      # Force push now
  %(prog)s push -m "Custom message"  # Force push with custom message
  %(prog)s milestone "Analysis done" # Mark milestone and push
  %(prog)s watch                     # Start file watcher mode
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show gateway status')
    
    # Start daemon command
    subparsers.add_parser('start', help='Start auto-push daemon')
    
    # Stop daemon command
    subparsers.add_parser('stop', help='Stop auto-push daemon')
    
    # Push command
    push_parser = subparsers.add_parser('push', help='Force push now')
    push_parser.add_argument('-m', '--message', help='Custom commit message')
    
    # Milestone command
    milestone_parser = subparsers.add_parser('milestone', help='Mark milestone and push')
    milestone_parser.add_argument('description', help='Milestone description')
    
    # Watch command
    subparsers.add_parser('watch', help='Start file watcher mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'status':
            status()
        elif args.command == 'start':
            start_daemon()
        elif args.command == 'stop':
            stop_daemon()
        elif args.command == 'push':
            force_push(args.message)
        elif args.command == 'milestone':
            milestone(args.description)
        elif args.command == 'watch':
            watch()
        else:
            parser.print_help()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()