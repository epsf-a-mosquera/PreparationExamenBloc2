#PreparationExamenBloc2/src/kafka_pipeline.py
"""
kafka_pipeline.py
-----------------
Launcher "tout-en-un" pour l'examen Bloc 2.

But :
- Lancer kafka_consumer + kafka_producer automatiquement
- Éviter d'ouvrir 2 terminaux

Fonctionnement :
1) Démarre le consumer dans un process séparé
2) Attend un court délai (le temps que le consumer soit prêt)
3) Lance le producer (qui envoie les messages)
4) Quand le producer termine, stoppe le consumer proprement (SIGINT)

Usage :
    python3 -m src.kafka_pipeline
"""

import os
import signal
import subprocess
import sys
import time
from typing import Optional


def start_module_process(module: str) -> subprocess.Popen:
    """
    Démarre un module Python avec `python -m <module>` dans un process séparé.
    Exemple : module="src.kafka_consumer"
    """
    cmd = [sys.executable, "-m", module]

    # Sur Linux/Mac : start_new_session=True permet de gérer proprement les signaux
    # (on pourra envoyer un SIGINT au process)
    return subprocess.Popen(
        cmd,
        stdout=None,   # laisse sortir les logs dans le terminal
        stderr=None,
        start_new_session=True,
        env=os.environ.copy(),
    )


def stop_process_gracefully(proc: subprocess.Popen, timeout_s: int = 10) -> None:
    """
    Tente d'arrêter un process proprement (SIGINT), puis force si besoin.
    """
    if proc.poll() is not None:
        return  # déjà arrêté

    try:
        # SIGINT = équivalent Ctrl+C -> laisse le code exécuter finally/close
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:
        # fallback : terminate
        proc.terminate()

    # Attendre une fermeture propre
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # si ça ne s'arrête pas, on force
        proc.kill()


def main(
    startup_wait_s: float = 2.0,
    stop_consumer_after_producer: bool = True,
) -> None:
    print("[PIPELINE] Démarrage consumer...")
    consumer_proc = start_module_process("src.kafka_consumer")

    try:
        # Laisser le consumer se connecter à Kafka et être prêt à lire
        time.sleep(startup_wait_s)

        print("[PIPELINE] Démarrage producer...")
        producer_proc = start_module_process("src.kafka_producer")

        # Attendre que le producer termine
        producer_rc = producer_proc.wait()
        print(f"[PIPELINE] Producer terminé (code retour={producer_rc}).")

        if stop_consumer_after_producer:
            print("[PIPELINE] Arrêt du consumer (SIGINT)...")
            stop_process_gracefully(consumer_proc, timeout_s=10)
            print("[PIPELINE] Consumer arrêté.")
        else:
            print("[PIPELINE] Consumer laissé actif. Ctrl+C pour arrêter.")
            # On attend indéfiniment
            consumer_proc.wait()

    except KeyboardInterrupt:
        print("\n[PIPELINE] Ctrl+C reçu -> arrêt des process...")
        stop_process_gracefully(consumer_proc, timeout_s=10)

    finally:
        # Sécurité : si consumer encore actif
        if consumer_proc.poll() is None:
            stop_process_gracefully(consumer_proc, timeout_s=5)


if __name__ == "__main__":
    main()
