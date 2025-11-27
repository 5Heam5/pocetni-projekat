"""
# Manifest
name: "Akademski Tim 2025 – Triple Threat Edition"
version: "2.7"
description: "Akademik (qwen2-vl) → Pisac (mistral-nemo) → Lektor (gemma3:27b-it-qat)"
author: "Ti + ja, Srbija 2025"
"""

import os
import json
import time
import re
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Union, List, Dict, Any

# Koristimo pydantic.Field za bolju integraciju sa OpenWebUI
class Valves(BaseModel):
    # Osnovna AI podešavanja
    OLLAMA_URL: str = Field(
        default="http://host.docker.internal:11434/v1", 
        description="URL Ollama API-ja",
        json_schema_extra={"group": "Osnovna podešavanja", "order": 1}
    )
    MODEL_AKADEMIK: str = Field(
        default="qwen2-vl:7b", 
        description="Model za analizu i izradu outline-a (Akademik)",
        json_schema_extra={"group": "Modeli", "order": 2}
    )
    MODEL_PISAC: str = Field(
        default="mistral-nemo:12b", 
        description="Model za pisanje prvog drafta (Pisac)",
        json_schema_extra={"group": "Modeli", "order": 3}
    )
    MODEL_LEKTOR: str = Field(
        default="gemma3:27b-it-qat", 
        description="Model za konačno lekturanje i formatiranje (Lektor)",
        json_schema_extra={"group": "Modeli", "order": 4}
    )
    TEMPERATURE: float = Field(
        default=0.3, 
        ge=0.0, le=1.0,
        description="Kreativnost modela (0.0 - 1.0)",
        json_schema_extra={"group": "Postavke modela", "order": 5}
    )
    MAX_TOKENS: int = Field(
        default=16000, 
        description="Maksimalan broj tokena za odgovor",
        json_schema_extra={"group": "Postavke modela", "order": 6}
    )
    MEMORY_TTL_SECONDS: int = Field(
        default=3600, 
        description="Vreme života memorije u sekundama",
        json_schema_extra={"group": "Napredna podešavanja", "order": 7}
    )
    MAX_TOKENS_PER_CHUNK: int = Field(
        default=8000, 
        description="Maksimalan broj tokena po delu za lekturanje",
        json_schema_extra={"group": "Napredna podešavanja", "order": 8}
    )

class Pipe:  # <--- KLJUČNA PROMENA: Koristi "Pipe" umesto "Tools"
    """
    Akademski Tim 2025 – Triple Threat Edition.
    Procesuiranje teksta kroz tri faze: Akademik (analiza), Pisac (draft), Lektor (final).
    """
    
    def __init__(self):
        self.valves = Valves()  # <--- Inicijalizuj valves koristeći Valves klasu
        self.client = AsyncOpenAI(base_url=self.valves.OLLAMA_URL, api_key="ollama")
        self.type = "manifold"
        self.memory = {}

    # OpenWebUI poziva ovu funkciju da dobavi listu dostupnih pipeline-ova
    def pipes(self):
        return [{"id": "triple_v2", "name": "Akademski Tim 2025 – Triple Threat v2"}]

    # OpenWebUI poziva ovu funkciju da dobije trenutne vrednosti ventila
    async def get_valves(self):
        return self.valves.dict()

    # OpenWebUI poziva ovu funkciju kada korisnik sačuva promene na ventilima
    async def update_valves(self, **valves):  # <--- Koristi **valves
        try:
            # Ažuriramo ventile jedan po jedan
            for key, value in valves.items():
                if hasattr(self.valves, key):
                    setattr(self.valves, key, value)
            
            # Ažuriramo klijenta sa novim URL-om ako je promenjen
            self.client = AsyncOpenAI(base_url=self.valves.OLLAMA_URL, api_key="ollama")
            
            return await self.get_valves()
        except Exception as e:
            raise Exception(f"Greška pri ažuriranju podešavanja: {str(e)}")

    # ... (OSTALE METODE - _cleanup_memory, _estimate_tokens, _split_text_into_chunks, _stream, pipe ostaju identične) ...
    # Ovde nastavite sa preostalim metodama iz vašeg originalnog koda koje nisu promenjene