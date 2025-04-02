uvicorn main:app --reload  || Ejecutar la API
uvicorn main:app --reload --port 8001  || puerto 8001

source venv/bin/activate  # Activar entorno vistual -> Mac

# Instalar dependencias
pip install fastapi uvicorn textblob python-multipart
python -m textblob.download_corpora  # Descargar datos de TextBlob
pip install transformers sentencepiece torch
pip install language-tool-python
pip install transformers torch accelerate
pip install --upgrade torch

brew install pkg-config
pkg-config: Este paquete ayuda a compilar programas que dependen de bibliotecas externas.

python3.11 -m venv venv || crear ambiente
source venv/bin/activate || ejecutar ambiente

# Instalar JAVA

brew install openjdk  # Instala OpenJDK
sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk  # Crea enlace simbólico

export JAVA_HOME=/opt/homebrew/opt/openjdk
export PATH="$JAVA_HOME/bin:$PATH"

source ~/.zshrc  # O source ~/.bashrc


http://localhost:8002/analyze-sentiment || POST

{
    "text": "i hate python"
}



http://localhost:8002/summarize-text || POST
{
    "text": "Python es un lenguaje de programación ampliamente utilizado en IA. Su sintaxis clara y su ecosistema de librerías como TensorFlow y PyTorch lo hacen ideal para principiantes y expertos. Además, cuenta con una comunidad activa que contribuye constantemente.",
    "max_length": 100,
    "min_length": 30
}

http://localhost:8002/check-grammar || POST

{
    "text": "Hazme sabe si tienes alguna pregunta. Yo iré a la tienda a comprar un lapiz."
}

http://localhost:8002/translate-text

{
    "text": "Hello, how are you?",
    "source_lang": "en",
    "target_lang": "es"
}



