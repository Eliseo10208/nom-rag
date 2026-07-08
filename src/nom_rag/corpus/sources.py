"""Registry of the corpus documents: STPS workplace-safety NOMs, official DOF text."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NomSource:
    nom_id: str  # full official id
    short_id: str  # used in chunk ids and citations
    title: str
    url: str
    # nota_detalle pages embed the norm as a nested document inside #DivDetalleNota;
    # normasOficiales pages are standalone exports.
    container: str | None


SOURCES: dict[str, NomSource] = {
    s.short_id: s
    for s in [
        NomSource(
            nom_id="NOM-035-STPS-2018",
            short_id="NOM-035",
            title="Factores de riesgo psicosocial en el trabajo — Identificación, análisis y prevención",
            url="https://www.dof.gob.mx/nota_detalle.php?codigo=5541828&fecha=23/10/2018",
            container="#DivDetalleNota",
        ),
        NomSource(
            nom_id="NOM-019-STPS-2011",
            short_id="NOM-019",
            title="Constitución, integración, organización y funcionamiento de las comisiones de seguridad e higiene",
            url="https://dof.gob.mx/nota_detalle.php?codigo=5185903&fecha=13/04/2011",
            container="#DivDetalleNota",
        ),
        NomSource(
            nom_id="NOM-017-STPS-2024",
            short_id="NOM-017",
            title="Equipo de protección personal — Selección, uso y manejo en los centros de trabajo",
            url="https://www.dof.gob.mx/normasOficiales/9496/stps/stps.html",
            container=None,
        ),
        NomSource(
            nom_id="NOM-030-STPS-2009",
            short_id="NOM-030",
            title="Servicios preventivos de seguridad y salud en el trabajo — Funciones y actividades",
            url="https://www.dof.gob.mx/normasOficiales/3923/stps/stps.htm",
            container=None,
        ),
    ]
}
