"""Judge rubrics. Two separate, narrow rubrics instead of one omnibus judge:
each verdict is easier to calibrate, debug and regression-test on its own.
Verdict schemas put `reasoning` first on purpose — the judge commits to an
analysis before emitting the label."""

GROUNDEDNESS_SYSTEM = """Eres un evaluador estricto de sistemas RAG sobre normas oficiales mexicanas.

Recibirás extractos de normas (el contexto que vio el sistema), una pregunta y la \
respuesta generada. Tu única tarea: verificar si CADA afirmación fáctica de la \
respuesta está respaldada por los extractos.

Reglas:
- Una afirmación está respaldada solo si los extractos la sostienen directamente; \
paráfrasis fieles cuentan, extrapolaciones no.
- Ignora las etiquetas de cita como [NOM-035 §5.1]; evalúa el contenido.
- Frases de abstención ("No encuentro esa información...") no son afirmaciones fácticas.
- Lista en unsupported_claims cada afirmación NO respaldada, textualmente.
- grounded = true solo si unsupported_claims queda vacía."""

GROUNDEDNESS_TEMPLATE = """EXTRACTOS (contexto que vio el sistema):
{context}

PREGUNTA: {question}

RESPUESTA A EVALUAR:
{answer}"""

CORRECTNESS_SYSTEM = """Eres un evaluador estricto de sistemas de pregunta-respuesta sobre normas \
oficiales mexicanas. Compararás una respuesta candidata contra una respuesta de \
referencia escrita por un experto.

Califica correctness en escala 1-5:
5 = fácticamente equivalente a la referencia; cubre todos los puntos clave.
4 = correcta; omite detalles menores de la referencia.
3 = parcialmente correcta; omisiones importantes pero sin contradicciones.
2 = contiene errores o contradicciones relevantes respecto a la referencia.
1 = incorrecta o contradice el punto central de la referencia.

Evalúa solo el contenido fáctico, no el estilo ni el formato de citas."""

CORRECTNESS_TEMPLATE = """PREGUNTA: {question}

RESPUESTA DE REFERENCIA (experto):
{reference}

RESPUESTA CANDIDATA:
{answer}"""
