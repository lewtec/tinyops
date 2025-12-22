# tinyops

Implementações de operações complexas puramente em `tinygrad`.

## Metodologia

Este é um projeto experimental. O objetivo é reconstruir algoritmos de domínios como visão computacional, áudio e machine learning utilizando estritamente as primitivas do `tinygrad`.

Ao evitar bibliotecas externas no runtime, garantimos que todo o fluxo de execução passe pelo grafo do `tinygrad`. Isso permite o uso agressivo de **kernel fusion**, onde múltiplas operações são compiladas em um único kernel de GPU/acelerador, maximizando a eficiência e portabilidade.

## Dependências

A única dependência de runtime é o **tinygrad**.

Bibliotecas como `numpy`, `opencv-python` e `scikit-learn` são dependências de desenvolvimento, utilizadas exclusivamente nos testes para verificar a paridade numérica e funcional das implementações.

## Desenvolvimento

Utilizamos `mise` para gerenciar o ambiente de desenvolvimento.

```bash
# Instalar dependências
mise install

# Rodar testes
mise run test
```
