#PreparationExamenBloc2/src/kafka_pipeline.py
"""
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
import threading  # ✅ Ajout : pour lire la sortie du consumer sans bloquer
from typing import Optional
from src.eco_impact import track_phase  # NEW : mesure producer Kafka



# ✅ Ajout : on garde une "trace" du dernier moment où le consumer a écrit quelque chose.
# L'idée : quand le producer a fini, on attend que le consumer soit "idle" (plus de logs) X secondes.
_last_consumer_activity_ts = 0.0


def _stream_consumer_output(proc: subprocess.Popen) -> None:
    """
    ✅ Ajout : Lit la sortie du consumer en continu pour :
    - l'afficher dans le terminal (comme avant)
    - mettre à jour _last_consumer_activity_ts pour détecter quand le consumer devient "idle"

    Important : on lance Python en mode non-bufferisé (-u) pour voir les logs tout de suite.
    """
    global _last_consumer_activity_ts

    if proc.stdout is None:
        return

    for line in proc.stdout:
        _last_consumer_activity_ts = time.time()
        # On ré-affiche exactement la sortie du consumer
        print(line, end="")


def start_module_process(module: str) -> subprocess.Popen:
    """
    Démarre un module Python avec `python -m <module>` dans un process séparé.
    Exemple : module="src.kafka_consumer"
    """
    # ✅ Modification MINIMALE :
    # -u : force un output non-bufferisé (sinon on ne peut pas détecter l'idle correctement)
    cmd = [sys.executable, "-u", "-m", module]

    # Sur Linux/Mac : start_new_session=True permet de gérer proprement les signaux
    # (on pourra envoyer un SIGINT au process)

    # ✅ Modification MINIMALE :
    # Pour le consumer : on capture stdout afin de détecter l'inactivité (idle)
    # Pour le producer : on laisse stdout=None comme avant (pas besoin de surveiller)
    capture_consumer_output = (module == "src.kafka_consumer")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if capture_consumer_output else None,
        stderr=subprocess.STDOUT if capture_consumer_output else None,
        text=True if capture_consumer_output else False,
        bufsize=1 if capture_consumer_output else -1,
        start_new_session=True,
        env=os.environ.copy(),
    )

    # ✅ Ajout : si on est sur le consumer, on démarre un thread qui affiche ses logs
    # et met à jour _last_consumer_activity_ts.
    if capture_consumer_output:
        global _last_consumer_activity_ts
        _last_consumer_activity_ts = time.time()  # activité "initiale"
        t = threading.Thread(target=_stream_consumer_output, args=(proc,), daemon=True)
        t.start()

    return proc


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
    # ✅ Ajout : paramètres "idle"
    consumer_idle_timeout_s: float = 3.0,
    max_wait_after_producer_s: float = 60.0,
) -> None:
    # NEW : mesure le temps d'émission des messages (CPU + I/O)
    with track_phase("kafka_pipeline_send"):
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
                # ✅ Modification MINIMALE :
                # On ne stoppe pas le consumer immédiatement.
                # On attend qu'il finisse de consommer ce qui reste (drain),
                # c.-à-d. qu'il devient "idle" pendant consumer_idle_timeout_s.
                print("[PIPELINE] Attente fin consommation (drain) avant arrêt consumer...")

                start_wait = time.time()
                while True:
                    # Si le consumer est déjà arrêté, on sort
                    if consumer_proc.poll() is not None:
                        break

                    # Condition 1 : consumer idle
                    idle_for = time.time() - _last_consumer_activity_ts
                    if idle_for >= consumer_idle_timeout_s:
                        print(f"[PIPELINE] Consumer idle depuis {idle_for:.1f}s -> arrêt.")
                        break

                    # Condition 2 : timeout de sécurité (évite d'attendre infiniment)
                    waited = time.time() - start_wait
                    if waited >= max_wait_after_producer_s:
                        print(f"[PIPELINE] Timeout drain ({waited:.1f}s) -> arrêt consumer.")
                        break

                    time.sleep(0.2)

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
