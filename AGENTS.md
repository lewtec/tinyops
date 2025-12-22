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
- **Validação de Kernels (CRÍTICO):** Todos os testes de operadores devem validar a continuidade e eficiência do kernel fusing. Para isso, use o decorator `@assert_one_kernel` (do `tinyops.test_utils`) em seus testes. O teste deve ser construído de forma que a operação sob teste, juntamente com a realização do resultado, gere **exatamente um kernel**.
    - **NÃO use fixtures** para preparação de dados. Use `@pytest.mark.parametrize` mesmo se tiver apenas um caso de teste.
    - Instancie e realize (`.realize()`) os tensores de entrada *antes* de definir a função interna decorada, ou garanta que eles sejam constantes que não geram kernels extras durante a execução da função.
    - O objetivo é garantir que não haja quebras no grafo de computação que impeçam o fusing completo da operação.

### Lidando com Kernel Fusing em Testes Complexos

**IMPORTANTE**: Quando a função sendo testada precisa de valores injetados (como valores random mockados ou tensors auxiliares), use `@pytest.mark.parametrize` ANTES do decorator `@assert_one_kernel`. Isso permite que os tensors sejam criados FORA do bloco medido.

**Padrão Recomendado:**
```python
@pytest.mark.parametrize("input_tensor,aux_values", [
    (
        Tensor(np.ones((10, 20), dtype=np.float32)).realize(),  # Criado antes do teste
        Tensor(np.array([0.5, 0.2], dtype=np.float32)),        # Valores auxiliares
    )
])
@assert_one_kernel
def test_operation(input_tensor, aux_values):
    # Agora apenas usa os parâmetros, sem criar novos tensors
    result = operation(input_tensor, _aux=aux_values)
    assert result.sum().item() > 0
```

**O que NÃO fazer:**
- ❌ Criar tensors dentro da função decorada com `@assert_one_kernel`
- ❌ Usar fixtures (são resolvidas dentro do bloco medido)
- ❌ Chamar `.realize()` dentro da função decorada (conta como kernel extra)
- ❌ Usar helper functions que criam tensors dentro do teste

**Limitação conhecida**: Para operações que usam `Tensor.arange()` ou `Tensor.ones()` internamente, pode ser necessário adicionar parâmetros opcionais (prefixados com `_`) para injetar esses tensors nos testes:
```python
def operation(x: Tensor, param: int, _indices: Tensor = None) -> Tensor:
    """Operação que precisa de índices.

    Args:
        x: Input tensor
        param: Parâmetro da operação
        _indices: (Internal) Pre-computed indices for testing
    """
    if _indices is None:
        _indices = Tensor.arange(x.shape[0])  # Gera kernel em produção
    # Usa _indices nas operações...
    return result
```

```python
# src/tinyops/stats/hist_test.py
import numpy as np
from tinygrad import Tensor
from tinyops.stats.hist import hist
from tinyops._core import assert_close
from tinyops.test_utils import assert_one_kernel
import pytest

# Use parametrize para inputs, evitando fixtures
@pytest.mark.parametrize("size, bins", [(100, 256)])
def test_hist(size, bins):
    # Setup: Cria e realiza inputs fora do escopo monitorado
    # Importante: Realizar inputs aqui garante que kernels de Load/Criação não contem
    data_np = np.random.randn(size).astype(np.float32)
    x = Tensor(data_np)
    x.realize()

    @assert_one_kernel
    def run_kernel():
        result = hist(x, bins=bins)
        result.realize() # Realiza resultado. O contador deve dar exatamente 1.
        return result

    # Executa a função decorada
    result = run_kernel()

    # Validação de valor
    expected = np.histogram(data_np, bins=bins)[0]
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
from tinyops.test_utils import assert_one_kernel
import pytest

@pytest.mark.parametrize("shape", [(100,)])
def test_median(shape):
    # Setup
    data_np = np.random.randn(*shape).astype(np.float32)
    data = Tensor(data_np)
    data.realize()

    @assert_one_kernel
    def run_median():
        result = median(data)
        result.realize()
        return result

    result = run_median()

    expected = np.median(data_np)
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
