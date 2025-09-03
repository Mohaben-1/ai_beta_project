from pydantic import BaseModel, Field
from typing import Any,Dict


class	FunctionDefinition(BaseModel):
	fn_name: str
	args_names: list[str]
	args_types: Dict[str, str]
	return_type: str

class	FunctionCall(BaseModel):
	prompt: str
	fn_name: str
	args: Dict[str, Any]
