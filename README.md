# Base de conocimiento de Club Tiempo Libre

La documentación se genera con [Zensical](https://zensical.org/) y se publica en GitHub Pages.

## Desarrollo local

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-docs.txt
zensical serve
```

La previsualización estará disponible en <http://localhost:8000/>.

## Validación

```sh
zensical build --clean --strict
```

Los cambios enviados a `main` se publican mediante GitHub Actions.

## Índice RAG

Cada cambio enviado a `main` también recrea el índice semántico de la documentación en Qdrant.