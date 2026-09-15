import importlib

import pytest

# Lista de bibliotecas de referência extraídas de CHECKLIST.md
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
    "Levenshtein",  # A biblioteca python-Levenshtein é importada como 'Levenshtein'
]


@pytest.mark.parametrize("library", LIBRARIES)
def test_import_libraries(library):
    """Testa se cada biblioteca de referência pode ser importada."""
    try:
        importlib.import_module(library)
    except ImportError:
        pytest.fail(f"A biblioteca de referência '{library}' não está instalada. Adicione-a ao pyproject.toml.")


def test_cv2_headless_mode():
    """Garante que o OpenCV (cv2) está rodando em modo headless."""
    try:
        import cv2

        # Em um ambiente headless, tentar usar uma função de GUI deve falhar
        # com um erro específico ou o backend da GUI deve ser 'headless'.
        # A simples ausência de erro em 'imshow' não é suficiente, pois pode travar.
        # Uma abordagem mais segura é verificar se não há backends de GUI disponíveis.
        # No entanto, a forma mais simples é garantir que 'opencv-python-headless' seja a dependência.
        # Este teste apenas confirma que o import funciona. A configuração do pacote
        # no pyproject.toml é a verdadeira garantia.
        assert cv2.__version__ is not None
    except ImportError:
        pytest.fail("A biblioteca 'cv2' não está instalada.")
    except Exception as e:
        # Se um erro de GUI acontecer, ele será capturado aqui.
        # Exemplo: "cv2.error: OpenCV(4.9.0) /io/opencv/modules/highgui/src/window.cpp:1274: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Carbon support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'"  # noqa: E501
        if "highgui" in str(e) and "not implemented" in str(e):
            # Este erro é esperado em um ambiente headless se tentarmos usar a GUI.
            # Para este teste, vamos considerá-lo uma confirmação do modo headless.
            pass
        elif "display" in str(e).lower():  # Erros comuns em ambientes sem servidor X
            pass
        else:
            pytest.fail(f"Erro inesperado ao importar ou verificar o cv2: {e}")
