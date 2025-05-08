"""
Script to run all tests.
"""
import os
import sys
import logging
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tests():
    """Run all tests in the tests directory."""
    try:
        # Get the tests directory
        tests_dir = Path(__file__).parent
        
        # Discover and run tests
        logger.info("Discovering tests...")
        test_suite = unittest.defaultTestLoader.discover(
            start_dir=str(tests_dir),
            pattern='test_*.py'
        )
        
        # Run tests
        logger.info("Running tests...")
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        
        # Log results
        logger.info(f"Tests completed. Ran {result.testsRun} tests.")
        if result.wasSuccessful():
            logger.info("All tests passed successfully!")
        else:
            logger.error("Some tests failed!")
            for error in result.errors:
                logger.error(f"Error: {error[0]}\n{error[1]}")
            for failure in result.failures:
                logger.error(f"Failure: {failure[0]}\n{failure[1]}")
        
        return result.wasSuccessful()
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        raise

def main():
    """Main function to run tests."""
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 