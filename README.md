# Color Adjustments

Progetto Python per ricolorare un'immagine mantenendo il contenuto originale. L'idea e':

1. ottenere un'immagine target ricolorata con un diffusion model locale o un altro strumento esterno;
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

## Refinement con residual mascherato

Quando la target contiene effetti locali che una singola LUT globale non puo' rappresentare bene, puoi usare `refine`. Il comando fitta comunque la LUT globale, poi applica una correzione residual solo nelle zone selezionate da una maschera morbida:

```bash
color-adjust refine \
  --input data/originale.jpg \
  --target outputs/target_ricolorata.png \
  --output outputs/ricolorata_refined.png \
  --lut-output outputs/ricolorata_lut.png \
  --lut outputs/trasformazione.cube \
  --mask outputs/residual_mask.png \
  --comparison outputs/refined_comparison.png \
  --report-json outputs/refined_metrics.json
```

La comparison del refine ha quattro pannelli: originale, target, risultato LUT e risultato refined. Il report JSON confronta le metriche della LUT pura con quelle del refined.

Parametri utili:

- `--residual-alpha`: intensita' della correzione residual;
- `--mask-threshold`: differenza colore minima prima di applicare la correzione;
- `--mask-softness`: morbidezza della soglia;
- `--mask-blur-radius`: sfocatura spaziale della maschera;
- `--max-residual`: limite massimo della correzione per canale RGB;
- `--positive-luma-only`: usa il residual solo dove la target diventa piu' luminosa.

Puoi attivare lo stesso passaggio direttamente nella pipeline:

```bash
color-adjust pipeline \
  --input data/originale.jpg \
  --target outputs/target_ricolorata.png \
  --output outputs/ricolorata_lut.png \
  --refine \
  --refined-output outputs/ricolorata_refined.png \
  --refined-mask outputs/residual_mask.png \
  --refined-comparison outputs/refined_comparison.png \
  --refined-report-json outputs/refined_metrics.json
```

## Uso con Diffusers

Backend disponibili:

- `instruct-pix2pix` / `instruct_pix2pix`: consigliato per istruzioni di editing, ad esempio cambiare globalmente il color grading;
- `img2img`: Stable Diffusion 1.5 image-to-image con `strength` basso di default (`0.25`), pensato per preservare contenuto e geometria;
- `sdxl`: SDXL image-to-image generico;
- `sdxl-turbo`: SDXL Turbo, piu' veloce ma con parametri diversi;
- `sd15`: Stable Diffusion 1.5 image-to-image.

Il backend predefinito e' `instruct-pix2pix`. Non serve ricordare il model id HuggingFace:

```bash
color-adjust pipeline \
  --input data/originale.jpg \
  --prompt "make the whole photograph warm golden hour, only change the color grading, preserve all objects and geometry" \
  --diffusion_backend instruct_pix2pix
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

Modalita' Stable Diffusion img2img a strength basso:

```bash
color-adjust pipeline \
  --input data/input.jpg \
  --prompt "cinematic warm sunset color grading, same scene, same objects, preserve details" \
  --negative_prompt "new objects, changed geometry, distorted shapes, extra details, different scene, different composition" \
  --diffusion_backend img2img \
  --strength 0.25 \
  --guidance_scale 7.5 \
  --num_inference_steps 30 \
  --seed 42
```

Equivalente con `python` dal repository:

```bash
.venv/bin/python -m color_adjust pipeline \
  --input data/input.jpg \
  --prompt "cinematic warm sunset color grading, same scene, same objects, preserve details" \
  --diffusion_backend img2img \
  --strength 0.25 \
  --guidance_scale 7.5 \
  --num_inference_steps 30 \
  --seed 42
```

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

- `cinematic warm sunset color grading, same scene, same objects, preserve details`
- `cold winter blue color grading, same scene, same objects, preserve structure`
- `teal and orange cinematic color grading, preserve the original content`
- `vintage film color grading, same composition, same objects`
- `high contrast dramatic color grading, preserve image structure`

Negative prompt consigliato:

- `new objects, changed geometry, distorted shapes, extra details, different scene, different composition`

## Eseguire prompt su un dataset

Lo script `scripts/run_dataset_prompts.py` lancia `color_adjust pipeline` su tutte le immagini di una cartella e su tutti i prompt presenti in un file di testo. Serve per testare in batch piu' prompt di ricolorazione sulle stesse immagini.

Uso base:

```bash
source .venv/bin/activate 
python scripts/run_dataset_prompts.py \
  --dataset-dir data/dataset \
  --prompts data/best.txt \
  --output-dir data/results_best \
  --backend instruct_pix2pix \
  --image-guidance-scale 1.2 \
  --seed 0
```

Il file passato con `--prompts` contiene un prompt per riga. Le righe vuote e quelle che iniziano con `#` vengono ignorate. Se una riga inizia con `negative:`, il testo dopo i due punti viene usato come negative prompt globale per tutte le esecuzioni.

Per ogni coppia immagine/prompt lo script crea una sottocartella dentro `--output-dir` e salva:

- target generata dal diffusion model: `*_target.png`;
- immagine finale ottenuta tramite LUT: `*_lut.png`;
- LUT in formato `.cube`: `*_lut.cube`;
- metriche PSNR/SSIM: `*_metrics.json`;
- confronto originale/target/LUT: `*_comparison.png`;
- manifest della singola esecuzione con prompt e comando usato: `*_run.json`.

Di default lo script cerca immagini ricorsivamente in `--dataset-dir` con estensioni comuni (`jpg`, `png`, `webp`, `bmp`, `tif`, `tiff`). Usa `--no-recursive` per leggere solo i file direttamente nella cartella indicata.

Opzioni utili:

- `--dry-run`: stampa i comandi e scrive i manifest senza generare immagini;
- `--keep-going`: continua con le altre immagini anche se una generazione fallisce;
- `--negative-prompt "..."`: sovrascrive l'eventuale riga `negative:` del file prompt;
- `--steps`, `--device`, `--max-side`, `--samples`, `--method`, `--luma-tolerance`: vengono inoltrate a `color_adjust pipeline`.

Per InstructPix2Pix, `--image-guidance-scale` controlla quanto il risultato resta vicino all'immagine originale: valori tipici da provare sono `1.0`, `1.2` e `1.5`.
