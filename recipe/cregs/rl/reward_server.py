import asyncio
import logging

import torch
import uvicorn
from fastapi import Body, FastAPI
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerBase

# Use logging module
logger = logging.getLogger(__name__)

def run_reward_server(model: AutoModelForTokenClassification, tokenizer: PreTrainedTokenizerBase, port: int):
    """
    Start the reward model service asynchronously.
    
    Args:
        model: Example: AutoModelForTokenClassification.from_pretrained(...)
        tokenizer: Corresponding tokenizer for the model.
        port: Port number for the service.
    """
    app = FastAPI()

    @app.post("/score")
    async def score(text: str = Body(..., embed=True)):
        # Run inference in a separate thread to keep the event loop non-blocking
        loop = asyncio.get_running_loop()
        
        def inference_fn():
            model.eval()
            device = model.device
            
            # Tokenize input text
            inputs = tokenizer(text, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # shape: (1, seq_len, num_labels)
                
                # Squeeze batch dimension
                logits = logits.squeeze(0)
                
                # If the model outputs a single scalar per token, flatten the last dimension
                if logits.shape[-1] == 1:
                    result = logits.squeeze(-1).tolist()
                else:
                    result = logits.tolist()
            return result

        try:
            scores = await loop.run_in_executor(None, inference_fn)
            return {"score": scores}
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return {"error": str(e)}

    logger.info(f"Starting reward server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
