# ----------------------------------------------------------
# Modules
# ----------------------------------------------------------

from typing import Literal, Dict, Any, List


# ----------------------------------------------------------
# Type Variables
# ----------------------------------------------------------

DeviceType = Literal["cpu", "cuda"]
ResponseType = Dict[str, Any]
OutputType = Dict[str, List[str]]
