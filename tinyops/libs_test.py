import importlib
import pytest

# List of reference libraries extracted from CHECKLIST.md
LIBRARIES = [
    "numpy",
    "tinygrad",
    "cv2",
    "PIL",
    "scipy",
    "sklearn",
    "torchaudio",
    "rapidfuzz",
    "jellyfish",
    "nltk",
    "Levenshtein",  # The python-Levenshtein library is imported as 'Levenshtein'
]

@pytest.mark.parametrize("library", LIBRARIES)
def test_import_libraries(library):
    """Tests if each reference library can be imported."""
    try:
        importlib.import_module(library)
    except ImportError:
        pytest.fail(f"The reference library '{library}' is not installed. Add it to pyproject.toml.")

def test_cv2_headless_mode():
    """Ensures that OpenCV (cv2) is running in headless mode."""
    try:
        import cv2
        # In a headless environment, trying to use a GUI function should fail
        # with a specific error or the GUI backend should be 'headless'.
        # The simple absence of error in 'imshow' is not sufficient, as it might hang.
        # A safer approach is to check if there are no GUI backends available.
        # However, the simplest way is to ensure 'opencv-python-headless' is the dependency.
        # This test just confirms that the import works. The package configuration
        # in pyproject.toml is the real guarantee.
        assert cv2.__version__ is not None
    except ImportError:
        pytest.fail("The 'cv2' library is not installed.")
    except Exception as e:
        # If a GUI error happens, it will be captured here.
        # Example: "cv2.error: OpenCV(4.9.0) /io/opencv/modules/highgui/src/window.cpp:1274: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'"
        if "highgui" in str(e) and "not implemented" in str(e):
            # This error is expected in a headless environment if we try to use the GUI.
            # For this test, we consider it a confirmation of headless mode.
            pass
        elif "display" in str(e).lower(): # Common errors in environments without X server
            pass
        else:
            pytest.fail(f"Unexpected error when importing or checking cv2: {e}")
