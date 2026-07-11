from __future__ import annotations

import torch
import torch.nn as nn

from app.utils.logger import logger

class BoundingBoxHead(nn.Module):
    """
    Predict normalized bounding boxes for each object query.
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
    
        super().__init__()
    
        self.embedding_dim = embedding_dim

        self.regressor = nn.Sequential(
        
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
        
            nn.ReLU(
                inplace=True,
            ),
        
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        
            nn.ReLU(
                inplace=True,
            ),
        
            nn.Linear(
                hidden_dim,
                4,
            ),
        
        )

        logger.info(
            "BoundingBoxHead initialized."
        )

    def forward(
        self,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict normalized bounding boxes.
    
        Parameters
        ----------
        queries
    
            (B,Q,C)
    
        Returns
        -------
        torch.Tensor
    
            (B,Q,4)
    
            (cx,cy,w,h)
        """
    
        boxes = self.regressor(
            queries,
        )
    
        return torch.sigmoid(
            boxes,
        )


    @property
    def output_dim(
        self,
    ) -> int:
        """
        Number of box coordinates.
        """
    
        return 4

    @staticmethod
    def to_xyxy(
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert normalized (cx, cy, w, h)
        boxes to (x1, y1, x2, y2).
        """
    
        cx, cy, w, h = boxes.unbind(
            dim=-1,
        )
    
        x1 = cx - (w / 2)
        y1 = cy - (h / 2)
    
        x2 = cx + (w / 2)
        y2 = cy + (h / 2)
    
        return torch.stack(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            dim=-1,
        )