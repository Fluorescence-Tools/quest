"""QuEst's service layer: named operations behind one call contract.

See :mod:`quest.backend.contract` for the identity, error codes and the
serialization boundary, :mod:`quest.backend.services` for the method table, and
:mod:`quest.backend.dispatcher` for the envelope and the in-process client.
"""

from .contract import (
    CONTRACT_VERSION,
    ERROR_CODES,
    PLUGIN_ID,
    contract_descriptor,
    error_code_for,
    service_error,
    service_success,
    to_payload,
)
from .dispatcher import InProcessClient, RemoteError, ServiceDispatcher
from .services import (
    METHODS,
    local_client,
    register_services,
    result_payload,
    simulation_payload,
)

__all__ = [
    "InProcessClient",
    "RemoteError",
    "ServiceDispatcher",
    "METHODS",
    "local_client",
    "register_services",
    "result_payload",
    "simulation_payload",
    "PLUGIN_ID",
    "CONTRACT_VERSION",
    "ERROR_CODES",
    "contract_descriptor",
    "error_code_for",
    "service_error",
    "service_success",
    "to_payload",
]
