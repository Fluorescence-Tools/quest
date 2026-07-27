"""QuEst's service layer: named operations behind one call contract.

See :mod:`quest.backend.services` for the method table and
:mod:`quest.backend.dispatcher` for the envelope and the in-process client.
"""

from .dispatcher import InProcessClient, RemoteError, ServiceDispatcher
from .services import METHODS, local_client, register_services, result_payload

__all__ = [
    "InProcessClient",
    "RemoteError",
    "ServiceDispatcher",
    "METHODS",
    "local_client",
    "register_services",
    "result_payload",
]
