# Despliegue en una VPS con Docker

Esta configuración está pensada para una VPS pequeña (1 vCPU, 4 GB RAM) y un
grupo de hasta ocho usuarios de uso moderado. Ejecuta una sola instancia de
FastAPI, conserva Chroma y las trazas en volúmenes, y publica únicamente Caddy
en los puertos 80/443.

## Arquitectura

```text
Internet
   |
   v
Caddy :80/:443  -- TLS + autenticación básica
   |
   v
FastAPI :8017   -- no se publica en el host
   |-- modelo e5 incluido en la imagen
   |-- volumen persistente: Chroma
   `-- volumen persistente: trazas
```

La imagen descarga el modelo y construye un índice semilla durante `docker
compose build`. En el primer arranque copia ese índice al volumen persistente.
Las reconstrucciones posteriores de la imagen no reemplazan un volumen que ya
contenga datos.

## 1. Preparar la VPS

Usa Ubuntu 24.04 y autentícate con una llave SSH. En el firewall del proveedor
permite únicamente:

- TCP 22 desde tus direcciones administrativas cuando sea posible.
- TCP 80 desde cualquier origen.
- TCP y UDP 443 desde cualquier origen.

No abras el puerto 8017. Docker puede publicar puertos saltándose algunas reglas
de UFW; por eso `compose.yaml` solo expone 8017 dentro de la red de Compose.

### Crear 2 GB de swap

En una VPS de 4 GB, el swap evita que una subida puntual de memoria termine el
proceso. No sustituye la RAM y no debe usarse como capacidad normal.

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

### Instalar Docker

Instala Docker Engine y el complemento de Compose desde el repositorio oficial:

<https://docs.docker.com/engine/install/ubuntu/>

Verifica la instalación:

```bash
sudo docker run --rm hello-world
sudo docker compose version
```

Los comandos restantes asumen un usuario autorizado para usar Docker. Recuerda
que pertenecer al grupo `docker` equivale prácticamente a acceso root.

## 2. Clonar y configurar

```bash
sudo mkdir -p /opt/nom-rag
sudo chown "$USER":"$USER" /opt/nom-rag
git clone https://github.com/Eliseo10208/nom-rag.git /opt/nom-rag
cd /opt/nom-rag
cp .env.vps.example .env
chmod 600 .env
```

Genera una contraseña larga para el grupo y conviértela a un hash aceptado por
Caddy. El comando solicita la contraseña de forma interactiva:

```bash
docker run --rm -it caddy:2-alpine caddy hash-password
```

Edita `.env` y completa:

- `GEMINI_API_KEY`: clave de Google AI Studio.
- `NOMRAG_DOMAIN`: dominio público, por ejemplo `nom.example.com`.
- `NOMRAG_AUTH_USER`: usuario compartido inicial.
- `NOMRAG_AUTH_HASH`: resultado completo del comando anterior. Déjalo entre
  comillas simples para que Compose no interprete los signos `$` del bcrypt.

Nunca subas `.env` a Git ni envíes la contraseña, la clave de Gemini o llaves
SSH privadas por chat.

## 3. Configurar DNS

Crea un registro `A` para `NOMRAG_DOMAIN` apuntando a la IPv4 de la VPS. Espera
a que resuelva antes de iniciar Caddy:

```bash
getent hosts nom.example.com
```

Caddy obtiene y renueva automáticamente el certificado TLS cuando el dominio
resuelve hacia la VPS y los puertos 80/443 son accesibles.

## 4. Construir y arrancar

```bash
cd /opt/nom-rag
docker compose config --quiet
docker compose build
docker compose up -d --wait --wait-timeout 240
docker compose ps
```

La primera compilación descarga PyTorch CPU y el modelo de embeddings; con un
solo vCPU tardará más que las actualizaciones normales. La aplicación no se
considera saludable hasta que el modelo y el índice hayan cargado.

Valida desde otra máquina:

```bash
curl -I https://nom.example.com
curl -u nomrag https://nom.example.com/api/health
```

La primera llamada debe devolver `401` sin credenciales. La segunda debe
devolver JSON con `"ok": true`.

## 5. Operación

### Consultar estado y consumo

```bash
docker compose ps
docker compose logs --tail=100 app
docker compose logs --tail=100 caddy
docker stats --no-stream
df -h
free -h
```

Con 4 GB de RAM, investiga si el uso sostenido supera 80 %, aparece actividad
continua de swap o las consultas se acumulan con frecuencia. Esas son señales
para subir al plan de 8 GB.

### Actualizar

El script solo acepta avances lineales de Git y espera a que el stack quede
saludable:

```bash
cd /opt/nom-rag
./ops/backup-volumes.sh
./ops/update.sh
```

Después de comprobar la actualización, las imágenes antiguas sin uso pueden
eliminarse con `docker image prune`.

### Respaldar datos de la aplicación

Hostinger ofrece respaldos de la VPS, pero conviene conservar también copias
portables de las trazas y Chroma:

```bash
cd /opt/nom-rag
./ops/backup-volumes.sh /ruta/a/un/directorio/respaldado
```

El script detiene brevemente FastAPI para obtener una copia consistente, crea
dos archivos `.tar.gz` y vuelve a iniciar la aplicación incluso si el respaldo
falla. Copia esos archivos fuera de la VPS.

## 6. Límites deliberados

- Un worker de Uvicorn: evita duplicar el modelo en RAM.
- Dos consultas RAG simultáneas: las demás esperan hasta 60 segundos.
- Límite de 3 GB para el contenedor de la aplicación.
- Logs de Docker rotados (tres archivos de 10 MB por servicio).
- Root filesystem de solo lectura; únicamente `/tmp`, Chroma, trazas y los
  directorios internos de Caddy son escribibles.
- Autenticación básica compartida para el grupo inicial. Si se necesitan
  usuarios individuales, revocación por persona o roles, debe reemplazarse por
  autenticación de aplicación.

## Solución rápida de problemas

```bash
# La imagen no termina de construir
docker compose build --progress=plain

# La aplicación no pasa el healthcheck
docker compose logs app

# Caddy no obtiene el certificado
docker compose logs caddy
getent hosts nom.example.com

# Ver el estado de los volúmenes
docker volume ls | grep nom-rag
```

No borres los volúmenes con `docker compose down -v`: `-v` elimina el índice,
las trazas y los certificados de Caddy.
