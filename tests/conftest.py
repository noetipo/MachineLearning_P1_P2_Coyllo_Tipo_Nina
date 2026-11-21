import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "app"))


@pytest.fixture(scope="session")
def raiz():
    return RAIZ


@pytest.fixture(scope="session")
def df():
    from churn.datos import cargar_datos
    return cargar_datos()


@pytest.fixture(scope="session")
def modelo():
    from churn.prediccion import cargar_modelo
    return cargar_modelo()


@pytest.fixture
def cliente_base():
    return {
        "genero": "Mujer", "adulto_mayor": "No", "tiene_pareja": "Sí", "tiene_dependientes": "No",
        "antiguedad_meses": 24, "servicio_telefonico": "Sí", "lineas_multiples": "No",
        "servicio_internet": "DSL", "seguridad_en_linea": "Sí", "respaldo_en_linea": "No",
        "proteccion_dispositivo": "No", "soporte_tecnico": "Sí", "tv_streaming": "No",
        "peliculas_streaming": "No", "tipo_contrato": "Un año", "factura_electronica": "No",
        "metodo_pago": "Transferencia bancaria (automática)", "cargo_mensual": 55.0,
    }
