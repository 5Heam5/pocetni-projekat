import os
import json
import time
import re
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Union, List, Dict, Any


class Pipeline:
    """
    Akademski Tim 2025 – Triple Threat Edition.
    Procesuiranje teksta kroz tri faze: Akademik (analiza), Pisac (draft), Lektor (final).
    """

    # Valves klasa mora biti definisana unutar Pipeline klase
    class Valves(BaseModel):
        # Osnovna AI podešavanja
        OLLAMA_URL: str = Field(
            default="http://host.docker.internal:11434/v1",
            description="URL Ollama API-ja",
            json_schema_extra={"group": "Osnovna podešavanja", "order": 1},
        )
        MODEL_AKADEMIK: str = Field(
            default="qwen2-vl:7b",
            description="Model za analizu i izradu outline-a (Akademik)",
            json_schema_extra={"group": "Modeli", "order": 2},
        )
        MODEL_PISAC: str = Field(
            default="mistral-nemo:12b",
            description="Model za pisanje prvog drafta (Pisac)",
            json_schema_extra={"group": "Modeli", "order": 3},
        )
        MODEL_LEKTOR: str = Field(
            default="gemma3:27b-it-qat",
            description="Model za konačno lekturanje i formatiranje (Lektor)",
            json_schema_extra={"group": "Modeli", "order": 4},
        )
        TEMPERATURE: float = Field(
            default=0.3,
            ge=0.0,
            le=1.0,
            description="Kreativnost modela (0.0 - 1.0)",
            json_schema_extra={"group": "Postavke modela", "order": 5},
        )
        MAX_TOKENS: int = Field(
            default=16000,
            description="Maksimalan broj tokena za odgovor",
            json_schema_extra={"group": "Postavke modela", "order": 6},
        )
        MEMORY_TTL_SECONDS: int = Field(
            default=3600,
            description="Vreme života memorije u sekundama",
            json_schema_extra={"group": "Napredna podešavanja", "order": 7},
        )
        MAX_TOKENS_PER_CHUNK: int = Field(
            default=8000,
            description="Maksimalan broj tokena po delu za lekturanje",
            json_schema_extra={"group": "Napredna podešavanja", "order": 8},
        )

    def __init__(self):
        # Ovde inicijalizujemo valves. OpenWebUI će automatski popuniti vrednosti
        # iz interfejsa kada korisnik sačuva podešavanja.
        self.valves = self.Valves()
        
        # Postavke za OpenWebUI
        self.type = "manifold"  # Ovaj pipeline rutira ka drugim modelima
        self.name = "Akademski Tim 2025" # Ime koje će biti prikazano

        # Inicijalizacija klijenta i memorije
        self.client = None # Inicijalizacija u update_valves
        self.memory = {}

    def pipes(self):
        """Vraća listu dostupnih pipeline-ova koje ovaj objekat nudi."""
        return [
            {
                "id": "triple_v2", 
                "name": "Akademski Tim 2025 – Triple Threat v2"
            }
        ]

    async def on_startup(self):
        """Ova metoda se poziva kada se pipeline učita."""
        # Ovde možete inicijalizati resurse koji su potrebni za ceo životni vek pipeline-a
        print(f"Pipeline '{self.name}' se pokreće...")
        # Inicijalizujemo klijenta sa početnim podešavanjima
        self.client = AsyncOpenAI(base_url=self.valves.OLLAMA_URL, api_key="ollama")

    async def on_shutdown(self):
        """Ova metoda se poziva kada se pipeline gasi."""
        print(f"Pipeline '{self.name}' se gasi...")
        # Ovde možete očistiti resurse
        pass

    async def on_valves_updated(self):
        """Ova metoda se poziva kada korisnik sačuva nova podešavanja (valves)."""
        # Ovo je ključno za primenu novih URL-ova ili modela bez restarta servisa
        print(f"Podešavanja za pipeline '{self.name}' su ažurirana.")
        self.client = AsyncOpenAI(base_url=self.valves.OLLAMA_URL, api_key="ollama")

    def _cleanup_memory(self):
        """Čišćenje stare memorije."""
        current_time = time.time()
        to_delete = [
            chat_id
            for chat_id, data in self.memory.items()
            if current_time - data.get("timestamp", 0) > self.valves.MEMORY_TTL_SECONDS
        ]
        for chat_id in to_delete:
            del self.memory[chat_id]

    def _estimate_tokens(self, text: str) -> int:
        """Gruba procena broja tokena (prosečno 4 karaktera po tokenu)."""
        return len(text) // 4

    def _split_text_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        """Deljenje teksta na manje delove (chunk-ove) na osnovu broja tokena."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            if current_tokens + para_tokens > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    async def _stream(self, model: str, prompt: Union[str, list[dict]], emitter):
        """Pomoćna funkcija za streamovanje odgovora od modela."""
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.valves.TEMPERATURE,
                max_tokens=self.valves.MAX_TOKENS,
                stream=True,
            )
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    await emitter({"type": "content", "content": content})
            return full_response
        except Exception as e:
            error_message = (
                f"\n\n[GRESKA] Komunikacija sa modelom '{model}' neuspesna: {str(e)}\n"
            )
            await emitter({"type": "content", "content": error_message})
            print(f"Greska u pipeline-u: {e}")
            return ""

    async def pipe(self, body: dict, __event_emitter__):
        """GLAVNA METODA - OpenWebUI poziva ovu metodu kada se pokrene pipeline"""
        self._cleanup_memory()

        # ─────────────── OSNOVNA VALIDACIJA ───────────────
        if not body.get("messages"):
            await __event_emitter__(
                {"type": "content", "content": "GRESKA: Nema poruka u zahtevu."}
            )
            return

        user_msg = body["messages"][-1]["content"] if body.get("messages") else ""
        if not user_msg or not str(user_msg).strip():
            await __event_emitter__(
                {
                    "type": "content",
                    "content": "GRESKA: Prazan tekst. Molimo unesite sadržajan upit.",
                }
            )
            return

        chat_id = body.get("chat_id", "default")
        images = []

        # Obrada slika iz poruka
        for msg in body.get("messages", []):
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image_url" and part.get(
                        "image_url", {}
                    ).get("url"):
                        images.append(part["image_url"]["url"])

        await __event_emitter__(
            {
                "type": "content",
                "content": "🚀 Pokrecem Akademski Tim 2025 – Triple Threat Edition v2...\n\n",
            }
        )

        # ────────────────────── 1. AKADEMIK ──────────────────────
        await __event_emitter__(
            {
                "type": "content",
                "content": "1️⃣ **AKADEMIK** – Analiza teme i izrada outline-a\n",
            }
        )
        akademik_content = [
            {
                "type": "text",
                "text": f"Napravi detaljan akademski plan i outline (strukturu) za temu:\n\n{user_msg}",
            }
        ]
        # Dodaj slike ako postoje
        if images:
            for img in images:
                akademik_content.append(
                    {"type": "image_url", "image_url": {"url": img}}
                )

        akademik_msg = [{"role": "user", "content": akademik_content}]

        outline = await self._stream(
            self.valves.MODEL_AKADEMIK, akademik_msg, __event_emitter__
        )
        if not outline.strip():
            await __event_emitter__(
                {
                    "type": "content",
                    "content": "\n\n❌ GRESKA: Akademik nije generisao outline. Proces se prekida.",
                }
            )
            return

        self.memory[chat_id] = {"outline": outline, "timestamp": time.time()}
        await __event_emitter__({"type": "content", "content": "\n\n---\n\n"})

        # ────────────────────── 2. PISAC ──────────────────────
        await __event_emitter__(
            {"type": "content", "content": "2️⃣ **PISAC** – Pisanje prvog drafta\n"}
        )
        pisac_prompt = f"""Piši kompletan akademski rad na srpskom jeziku.
Striktno prati sledeci outline:
{outline}

Zahtevi:
- Originalan, naucni stil.
- Koristi APA 7 stil citiranja (ne izmišljaj reference, ako nemaš informacije, koristi opšte formate).
- Ciljna dužina: 2500–4000 reci.
- Fokusiraj se na sadržaj, ne na formatiranje."""

        draft = await self._stream(
            self.valves.MODEL_PISAC, pisac_prompt, __event_emitter__
        )
        if not draft.strip():
            await __event_emitter__(
                {
                    "type": "content",
                    "content": "\n\n❌ GRESKA: Pisac nije generisao draft. Proces se prekida.",
                }
            )
            return

        self.memory[chat_id]["draft"] = draft
        self.memory[chat_id]["timestamp"] = time.time()
        await __event_emitter__({"type": "content", "content": "\n\n---\n\n"})

        # ────────────────────── 3. LEKTOR ──────────────────────
        await __event_emitter__(
            {
                "type": "content",
                "content": "3️⃣ **LEKTOR** – Finalna obrada, lekturanje i formatiranje\n",
            }
        )

        draft_tokens = self._estimate_tokens(draft)
        await __event_emitter__(
            {
                "type": "content",
                "content": f"[INFO] Draft ima ~{draft_tokens} tokena. Proveravam da li je potrebno deljenje...\n",
            }
        )

        if draft_tokens > self.valves.MAX_TOKENS_PER_CHUNK:
            await __event_emitter__(
                {
                    "type": "content",
                    "content": f"✂️ Tekst je predugacak, delimo ga na manje delove za lekturanje...\n",
                }
            )
            final_parts = []
            chunks = self._split_text_into_chunks(
                draft, self.valves.MAX_TOKENS_PER_CHUNK
            )

            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                await __event_emitter__(
                    {
                        "type": "content",
                        "content": f"\n🔨 Obradjujem deo {i+1}/{len(chunks)}...\n",
                    }
                )

                if is_last:
                    lektor_prompt = f"""Ti si konačni lektor i urednik.
Zadatak za POSLEDNJI deo teksta:
1. Popravi gramatiku, stil, ponavljanja.
2. Osiguraj 100% srpski jezik.
3. DODAJ na sam pocetak celog rada (ne samo ovog dela) sažetak (Abstract) i ključne reci.
4. Na sam kraj celog rada (ne samo ovog dela) dodaj formatiranu Literaturu prema APA 7 standardu.
5. Vrati ISKLJUCIVO obrađen deo teksta.

Deo za lekturanje:
{chunk}"""
                else:
                    lektor_prompt = f"""Ti si lektor i urednik.
Zadatak za SREDNJI deo teksta:
1. Popravi gramatiku, stil, ponavljanja.
2. Osiguraj 100% srpski jezik.
3. NE DODAVAJ sažetak, ključne reci ili literaturu. To ce biti urađeno na kraju.
4. Vrati ISKLJUCIVO obrađen deo teksta.

Deo za lekturanje:
{chunk}"""

                final_part = await self._stream(
                    self.valves.MODEL_LEKTOR, lektor_prompt, __event_emitter__
                )
                if final_part:
                    final_parts.append(final_part)

            final = "\n\n".join(final_parts)

        else:
            lektor_prompt = f"""Ti si konačni lektor i urednik.
Zadatak:
1. Popravi gramatiku, stil, ponavljanja u celom tekstu.
2. Formatiraj naslove i citate prema APA 7 standardu.
3. Osigurati 100% srpski jezik.
4. Dodati na pocetak sažetak (Abstract) i ključne reci.
5. Dodati na kraj formatiranu Literaturu.
6. Vrati konačan rad spreman za predaju.

Draft:
{draft}"""
            final = await self._stream(
                self.valves.MODEL_LEKTOR, lektor_prompt, __event_emitter__
            )

        await __event_emitter__(
            {
                "type": "content",
                "content": f"\n\n🎉 FINALNI MASTER RAD JE SPREMAN!\n\n{final}",
            }
        )

        # Čišćenje memorije za završen chat
        if chat_id in self.memory:
            del self.memory[chat_id]
