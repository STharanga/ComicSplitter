#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from gui.main_window import MainWindow

def main():
    """Main entry point for the Comic Splitter application."""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('comic_splitter.log')
            ]
        )
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Comic Splitter application")
        
        # Create and run main window
        app = MainWindow()
        try:
            app.run()
        finally:
            app.cleanup()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()