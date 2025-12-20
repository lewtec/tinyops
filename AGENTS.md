# AGENTS.md

## Objetivo
tinyops é uma biblioteca de operações implementadas puramente em tinygrad. O objetivo é gerar kernels fusionados e otimizados que podem ser exportados para outras linguagens/runtimes.
A restrição de usar apenas tinygrad para implementações (com libs externas apenas para testes de conformidade) garante que todo o grafo de computação passa pelo sistema de kernel fusion do tinygrad.

**Funções são stateless.** Não há train loops, classes com estado, ou interface fit/predict. O usuário monta o loop e gerencia estado. Funções recebem dados e retornam resultados.

## Setup
```bash
# instala mise se não tiver
if ! command -v mise &> /dev/null; then
    curl https://mise.run | sh
fi

# mise instala uv, uv instala deps
mise install
uv sync
```

## Estrutura
```
src/tinyops/
├── _core/          # helpers internos (tipos, tolerâncias, validação)
├── linalg/         # álgebra linear
├── stats/          # estatísticas, histogramas
├── image/          # transformações de imagem
├── audio/          # processamento de áudio
├── signal/         # processamento de sinais
├── io/             # encoder/decoder de arquivos, se implementável em tinygrad (wav, bmp)
└── ml/             # algoritmos de machine learning (sklearn-like)
```

Cada função em arquivo próprio com teste colocalizado:
```
modulo/
├── __init__.py
├── func.py
└── func_test.py
```

## Implementação
- Única dependência runtime: `tinygrad`
- Recebe/retorna `tinygrad.Tensor`
- Type hints obrigatórios
- Sem `**kwargs`
- Paths usam `pathlib`

```python
# src/tinyops/stats/hist.py
from tinygrad import Tensor

def hist(x: Tensor, bins: int) -> Tensor:
    """Calcula histograma do tensor."""
    ...
```

## Testes
- Arquivo `*_test.py` ao lado da implementação
- Compara output com lib original (numpy, cv2, torch*, etc)
- Usa helper `_core.assert_close` pra tolerância

```python
# src/tinyops/stats/hist_test.py
import numpy as np
from tinyops.stats.hist import hist
from tinyops._core import assert_close

def test_hist():
    x = ...  # dados de teste
    result = hist(x, bins=256)
    expected = np.histogram(x.numpy(), bins=256)[0]
    assert_close(result, expected)
```

## Adicionar nova função
1. Escolhe próxima função não marcada no `CHECKLIST.md`
2. Consulta documentação da lib original pra entender comportamento esperado
3. Cria arquivo em `src/tinyops/{modulo}/{func}.py`
4. Implementa função usando apenas `tinygrad`
5. Cria `src/tinyops/{modulo}/{func}_test.py` comparando com lib original
6. Importa no `__init__.py` do módulo
7. Roda `mise run test -- -k {func}` pra validar
8. Marca como `[x]` no `CHECKLIST.md`

Exemplo para `stats.median`:

```bash
# 1. implementação
touch src/tinyops/stats/median.py
```

```python
# src/tinyops/stats/median.py
from tinygrad import Tensor

def median(x: Tensor, axis: int | None = None) -> Tensor:
    """Retorna a mediana ao longo do eixo."""
    ...
```

```python
# src/tinyops/stats/median_test.py
import numpy as np
from tinygrad import Tensor
from tinyops.stats.median import median
from tinyops._core import assert_close

def test_median():
    data = np.random.randn(100).astype(np.float32)
    result = median(Tensor(data))
    expected = np.median(data)
    assert_close(result, expected)
```

```python
# src/tinyops/stats/__init__.py
from .median import median
```

## Comandos
```bash
mise run test           # roda todos os testes
mise run test -- -k hist  # testa só hist
```
