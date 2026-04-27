# Color Adjustments

Progetto Python per ricolorare un'immagine mantenendo il contenuto originale. L'idea e':

1. ottenere un'immagine target ricolorata con un diffusion model, ChatGPT, Nanobanana o un altro strumento;
2. campionare coppie di pixel corrispondenti tra immagine originale e target;
3. fittare una 3D LUT che approssima la trasformazione cromatica;
4. applicare la LUT all'immagine originale;
5. valutare il risultato con PSNR e SSIM rispetto al target;
6. salvare un'immagine di comparazione tra originale, target/diffusion e risultato LUT.

La parte diffusion e' opzionale: la pipeline principale funziona anche passando direttamente una target image gia' generata.

## Installazione

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Per usare anche HuggingFace Diffusers:

```bash
pip install -e ".[diffusion]"
```

### Diffusers con AMD ROCm

Su AMD non usare direttamente `.[diffusion]` come primo comando: quella extra installa `torch` dal normale indice pip e potrebbe non installare una build ROCm. Installa prima PyTorch ROCm seguendo il comando aggiornato dal selettore ufficiale PyTorch, scegliendo Linux, Pip, Python e ROCm:


Comandi per ROCm 7:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2
pip install -e ".[diffusion-rocm]"
pip install accelerate
```

Poi verifica che PyTorch veda la GPU AMD:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print('hip', torch.version.hip); print('gpu', torch.cuda.is_available())"
```

PyTorch usa ancora l'API `torch.cuda` anche con backend ROCm.

## Uso rapido con target gia' generata

Genera o prepara una target ricolorata con lo stesso contenuto dell'immagine originale, poi esegui:

```bash
color-adjust fit \
  --input data/originale.jpg \
  --target data/target_diffusion.png \
  --output outputs/ricolorata_lut.png \
  --lut outputs/trasformazione.cube \
  --report-json outputs/metriche.json
```

Il comando salva l'immagine ricolorata tramite LUT, la LUT in formato `.cube`, il report metriche JSON e una comparison image con tre pannelli: originale, target/diffusion e LUT applicata all'originale. Stampa anche PSNR/SSIM rispetto alla target.

Se non passi `--output`, il comando crea automaticamente una cartella nel punto da cui lanci il comando, usando il nome dell'immagine di input. Per esempio:

```bash
color-adjust fit \
  --input data/inputs/street-1.jpg \
  --target outputs/target_grade_warm.png
```

crea:

```text
./street-1/street-1_from_target_grade_warm_lut.png
./street-1/street-1_from_target_grade_warm_lut.cube
./street-1/street-1_from_target_grade_warm_lut_metrics.json
./street-1/street-1_from_target_grade_warm_lut_comparison.png
```

La stessa logica vale per `grade`, `diffuse`, `pipeline` e `apply`: i path espliciti hanno sempre priorita', ma se li ometti vengono generati automaticamente. Per `fit` e `pipeline` puoi anche passare `--comparison` per scegliere dove salvare la comparison image.

## Target cromatica deterministica

Per verificare che la LUT funzioni senza introdurre errori dovuti al diffusion model, puoi creare una target sintetica di color grading:

```bash
color-adjust grade \
  --input data/originale.jpg \
  --style warm-sunset \
  --strength 1.0 \
  --output outputs/target_warm.png

color-adjust fit \
  --input data/originale.jpg \
  --target outputs/target_warm.png \
  --output outputs/ricolorata_warm_lut.png \
  --lut outputs/warm.cube
```

Stili disponibili: `warm-sunset`, `cool-winter`, `teal-orange`, `vintage`, `autumn`, `high-contrast`.

Quando la target viene da un diffusion model e cambia troppo ombre o dettagli locali, puoi provare a filtrare campioni poco coerenti:

```bash
color-adjust fit \
  --input data/originale.jpg \
  --target outputs/target_diffusion.png \
  --output outputs/ricolorata_lut.png \
  --luma-tolerance 0.18
```

## Uso con Diffusers

Backend disponibili:

- `instruct-pix2pix`: consigliato per istruzioni di editing, ad esempio cambiare globalmente il color grading;
- `sdxl`: SDXL image-to-image generico;
- `sdxl-turbo`: SDXL Turbo, piu' veloce ma con parametri diversi;
- `sd15`: Stable Diffusion 1.5 image-to-image.

Il backend predefinito e' `instruct-pix2pix`. Non serve ricordare il model id HuggingFace:

```bash
color-adjust pipeline \
  --input data/originale.jpg \
  --prompt "make the whole photograph warm golden hour, only change the color grading, preserve all objects and geometry" \
  --backend instruct-pix2pix
```

Per InstructPix2Pix, `--strength` non viene usato. Il parametro piu' importante e':

```bash
--image-guidance-scale 1.2
```

Valori piu' alti preservano di piu' l'immagine originale, valori piu' bassi permettono cambiamenti piu' forti. In pratica prova `1.0`, `1.2`, `1.5`.

Se ometti i path di output, `pipeline` salva automaticamente tutti i file utili nella cartella dell'immagine:

```text
./street-1/street-1_diffusion_instruct-pix2pix_igs120_seed0_target.png
./street-1/street-1_diffusion_instruct-pix2pix_igs120_seed0_lut.png
./street-1/street-1_diffusion_instruct-pix2pix_igs120_seed0_lut.cube
./street-1/street-1_diffusion_instruct-pix2pix_igs120_seed0_lut_metrics.json
./street-1/street-1_diffusion_instruct-pix2pix_igs120_seed0_lut_comparison.png
```

La comparison image affianca originale, target diffusion e immagine finale ottenuta applicando la LUT all'originale.

```bash
color-adjust pipeline \
  --input data/originale.jpg \
  --prompt "same scene, cinematic teal and orange color grading, preserve all objects" \
  --backend sdxl \
  --generated-target outputs/target_diffusion.png \
  --output outputs/ricolorata_lut.png \
  --lut outputs/trasformazione.cube \
  --strength 0.35 \
  --steps 30
```

La generazione image-to-image puo' modificare anche forma e dettagli. Per questo progetto conviene usare prompt che chiedono esplicitamente di conservare oggetti, composizione e geometria, e tenere `--strength` basso o medio.

`--model-id` resta disponibile come override avanzato. Se viene passato senza `--backend`, il programma prova a dedurre il backend dal nome del modello.

## Applicare una LUT salvata

```bash
color-adjust apply \
  --input data/nuova_immagine.jpg \
  --lut outputs/trasformazione.cube \
  --output outputs/nuova_ricolorata.png
```

## Prompt consigliati

Esempi:

- `same image content, warm sunset color grading, preserve geometry and objects`
- `same scene, cold blue winter color palette, preserve all details`
- `same photo, vintage film colors, faded highlights, preserve composition`
- `same image, high contrast black and gold color grading, preserve objects`

