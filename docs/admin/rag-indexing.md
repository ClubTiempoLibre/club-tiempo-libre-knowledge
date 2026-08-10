# Indexación RAG en Qdrant

El repositorio Git es la fuente de verdad de la base de conocimiento. El workflow `.github/workflows/index-qdrant.yml` reconstruye la colección de Qdrant desde el contenido versionado cada vez que se envía un cambio a `main`.

## Arquitectura

El proceso ejecuta estos pasos:

1. Hace checkout del commit de `main`.
2. Descubre los documentos Markdown y MDX válidos.
3. Divide cada documento siguiendo sus encabezados `#`, `##` y `###`.
4. Genera embeddings densos con `BAAI/bge-m3` mediante Sentence Transformers.
5. Elimina, crea y rellena de nuevo la colección `club-tiempo-libre-knowledge`.
6. Comprueba que Qdrant contiene exactamente el número esperado de vectores.

Las ejecuciones se serializan para que dos pushes cercanos no escriban la colección simultáneamente. La reconstrucción completa evita conservar chunks de documentos eliminados o renombrados.

## Descubrimiento y chunking

El indexador recorre el repositorio y acepta archivos `.md` y `.mdx`. Excluye:

- `.git`, `.github` y cualquier fichero o directorio oculto.
- `site`, `_site`, `dist` y `build`.
- `node_modules`, `.venv`, `venv` y `__pycache__`.
- El `README.md` técnico de la raíz.
- Esta página administrativa, para que las instrucciones del pipeline no contaminen el conocimiento del Club Tiempo Libre.

Cada chunk corresponde preferentemente al contenido situado bajo un encabezado. El texto enviado al modelo empieza con la jerarquía completa, por ejemplo:

```md
# Campamentos

## Inscripciones

### Pagos

Contenido de la subsección...
```

Una sección de hasta 1.800 palabras se conserva íntegra. Las secciones mayores se dividen primero por párrafos y, solo cuando es imprescindible, por líneas o palabras. Las listas, tablas y enlaces Markdown se mantienen siempre que el límite lo permite. Los encabezados que aparezcan dentro de bloques de código no modifican la jerarquía.

Los identificadores son UUID v5 deterministas derivados de la ruta, la jerarquía, la repetición del encabezado y el número de parte. Una nueva ejecución sobre el mismo documento produce los mismos `chunk_id`.

## Metadatos

Cada punto de Qdrant contiene:

| Campo | Contenido |
| --- | --- |
| `content` | Texto del chunk con el contexto de encabezados. |
| `source_path` | Ruta relativa del documento en el repositorio. |
| `file_name` | Nombre del fichero Markdown. |
| `title` | Primer encabezado `#`, o el nombre del fichero como fallback. |
| `section` | Encabezado `##` activo. |
| `subsection` | Encabezado `###` activo. |
| `heading_path` | Jerarquía legible separada por ` > `. |
| `repo` | `club-tiempo-libre-knowledge`. |
| `indexed_at` | Fecha y hora UTC de la ejecución. |
| `chunk_id` | Identificador estable del chunk. |

`BAAI/bge-m3` genera vectores de 1.024 dimensiones. La colección usa distancia Cosine. Un cliente que genere embeddings para consultas debe utilizar el mismo modelo y normalización para obtener resultados comparables.

## Configuración de GitHub Actions

En **Settings > Secrets and variables > Actions** del repositorio, configura:

1. En **Variables**, crea `QDRANT_URL` con la URL HTTPS pública de Qdrant, sin barra final. No es un secreto.
2. En **Secrets**, crea `QDRANT_API_KEY` con la API key de Qdrant.

No guardes estos valores en Git. El runner alojado por GitHub debe poder alcanzar `QDRANT_URL`. Si se usa Cloudflare Access, la protección interactiva debe limitarse al dashboard; los endpoints de API utilizados por el indexador deben admitir la API key de Qdrant.

El workflow falla de forma explícita si falta una variable, si no encuentra Markdown válido, si Qdrant no responde, si el modelo devuelve una dimensión inesperada o si el recuento final no coincide.

## Prueba local

Prepara el entorno desde la raíz del repositorio:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Prueba descubrimiento, chunking e IDs sin descargar el modelo ni modificar Qdrant:

```sh
python -m unittest discover -s tests -v
python scripts/index_qdrant.py --dry-run
```

Para ejecutar la indexación completa, define temporalmente las variables:

```sh
export QDRANT_URL="https://qdrant.example.com"
read -s QDRANT_API_KEY
export QDRANT_API_KEY
python scripts/index_qdrant.py
unset QDRANT_API_KEY
```

Variables opcionales:

| Variable | Valor predeterminado | Uso |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Modelo de Sentence Transformers. |
| `EMBEDDING_BATCH_SIZE` | `2` | Textos procesados por lote. |
| `QDRANT_UPLOAD_BATCH_SIZE` | `64` | Puntos enviados por petición. |
| `QDRANT_TIMEOUT_SECONDS` | `60` | Timeout de conexión y peticiones. |

## Ejemplo de Qdrant en Raspberry Pi

Este Compose utiliza almacenamiento persistente y exige una API key. Los puertos se vinculan solo a loopback; un proxy inverso o un túnel puede alcanzar el contenedor mediante una red Docker compartida.

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    restart: always
    container_name: qdrant
    ports:
      - "127.0.0.1:6333:6333"
      - "127.0.0.1:6334:6334"
    expose:
      - 6333
      - 6334
      - 6335
    configs:
      - source: qdrant_config
        target: /qdrant/config/production.yaml
    environment:
      QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY}
    volumes:
      - ./qdrant_data:/qdrant/storage

configs:
  qdrant_config:
    content: |
      log_level: INFO
```

Guarda la clave junto al Compose en un `.env` que no se versionará:

```sh
umask 077
printf 'QDRANT_API_KEY=%s\n' "$(openssl rand -hex 32)" > .env
docker compose up -d
```

Antes de exponer Qdrant, limita el acceso de red, configura TLS en el proxy o túnel y verifica que `/collections` devuelve `401` sin la cabecera `api-key`.