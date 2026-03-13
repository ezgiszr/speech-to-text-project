import io
import time
import logging
import psutil
from faster_whisper import WhisperModel

# --- Log Ayarları ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("faster_whisper").setLevel(logging.INFO)


def load_transcription_model(model_size="distil-large-v3"):
    try:
        logger.info(f"Model yükleniyor: '{model_size}' (CPU, int8 kuantalama)")
        cpu_threads = max(1, psutil.cpu_count(logical=True) - 1)

        model = WhisperModel(
            model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads
        )
        logger.info("Model başarıyla yüklendi ve kullanıma hazır.")
        return model
    except Exception as e:
        logger.error(f"Model yüklenirken kritik hata: {e}")
        raise


def transcribe_audio(audio_path, model):

    try:
        logger.info(f"İşleniyor: {audio_path}")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        baslangic_zamani = time.time()

        segments, info = model.transcribe(
            audio_path,
            language="en",
            condition_on_previous_text=False,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(  # sessizlik ve gürültü filtreleme sistemi. Halüsinasyonu azaltmak için.
                min_silence_duration_ms=500,
                min_speech_duration_ms=250,
                # max_speech_duration_s=30  konuşmayı chunklara bölmek için
            ),
        )

        parcalar = []
        for segment in segments:
            parcalar.append(segment.text.strip())

        tam_metin = " ".join(parcalar)

        gecen_sure = time.time() - baslangic_zamani
        logger.info(f"Çeviri tamamlandı! Geçen süre: {gecen_sure:.2f} saniye.")

        return tam_metin.strip()

    except Exception as e:
        logger.error(f"Transkripsiyon sırasında hata oluştu: {e}")
        raise


# ==========================================
# ANA ÇALIŞTIRMA BLOĞU
# ==========================================
if __name__ == "__main__":
    
    audio_file = "harvard.wav"
    output_file = "transcript.txt"

    # 1. Modeli yükleS
    sistem_modeli = load_transcription_model()

    # 2. Sesi metne çevir
    elde_edilen_metin = transcribe_audio(audio_file, sistem_modeli)

    # 3. Elde edilen metni txt dosyasına kaydet
    logger.info(f"Metin '{output_file}' dosyasına kaydediliyor...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(elde_edilen_metin)

    logger.info("İşlem başarıyla tamamlandı!")  
