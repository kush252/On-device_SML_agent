from pydantic import BaseModel, Field
from typing import Dict, Any

class CategorySelection(BaseModel):
    """Schema for the SLM to select the top-level category."""
    category: str = Field(
        description="The selected category: 'browser', 'application', or 'interaction'."
    )
    reasoning: str = Field(
        description="A short explanation of why this category was chosen based on the request."
    )

class ToolSelection(BaseModel):
    """Schema for the SLM to select the exact tool and its arguments."""
    tool_name: str = Field(
        description="The exact name of the tool to execute from the registry."
    )
    arguments: Dict[str, Any] = Field(
        description="A dictionary of arguments to pass to the tool.",
        default_factory=dict
    )
    reasoning: str = Field(
        description="A short explanation of why this tool and these arguments were chosen."
    )
