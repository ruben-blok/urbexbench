# UrbexBench

Een benchmark (toets) om te testen hoe goed verschillende AI‑modellen kunnen classificeren of een locatie in satellietbeelden verlaten is of niet. Dit kan lastig zijn voor een AI‑model, omdat het visuele aanwijzingen moet herkennen die kunnen duiden op een verlaten locatie, zoals begroeiing of kapotte daken. Er zitten ook afbeeldingen bij die niet verlaten zijn; dan moet het model dingen herkennen zoals een bouwplaats, een nieuw dak of mooi bijgewerkt gras.

### Voorbeeldfoto die beoordeeld wordt:
![](img/abandoned/50.397637_4.498408.png)

## Score

| Rang | Model | Parameters | Nauwkeurigheid | Totaal |
|----- |-------------------------------|------|----|-----|
| 1🥇  | Gemini 3.7 Flash              | -    | 149 | 200 |
| 2🥈  | Gemini 3.1 Flash Lite Preview | -    | 142 | 200 |
| 3🥉  | Gemini 3.1 Flash Lite         | -    | 140 | 200 |
| 4    | Qwen3 VL                      | 30   | 138 | 200 |
| 5    | DeepSeek V4 Flash Vision      | -    | 137 | 200 |
| 6    | Gemini 2.5 Flash Lite         | -    | 136 | 200 |
| 6    | Qwen3.5 35B-A3B               | 35   | 136 | 200 |
| 8    | Ox Alpha                      | -    | 133 | 200 |
| 8    | Qwen3 VL                      | 32   | 133 | 200 |
| 8    | Qwen3.5                       | 27   | 133 | 200 |
| 11   | Nemotron 3 Nano Omni          | 30   | 132 | 200 |
| 12   | Nemotron Nano 12B V2 VL       | 12   | 131 | 200 |
| 12   | MiMo V2.5                     | -    | 131 | 200 |
| 14   | Gemma 3                       | 27   | 129 | 200 |
| 14   | Gemma 3                       | 12   | 129 | 200 |
| 14   | Qwen3.5                       | 9    | 129 | 200 |
| 17   | Qwen3 VL                      | 8    | 128 | 200 |
| 18   | Gemma 3                       | 4    | 125 | 200 |
| 18   | Gemma 4                       | 26   | 125 | 200 |
| 20   | Gemma 4                       | 31   | 119 | 200 |

\* De evaluaties draaien zonder reasoning waar dat mogelijk is. Gemini 3.7 Flash en Ox Alpha verplichten reasoning op hun endpoint en zijn daarom getest op de laagste ondersteunde stand (`low` / `minimal`).

## Werking

1. **Afbeelding coderen**: Afbeeldingen worden geconverteerd naar base64 en naar de API gestuurd
2. **Prompt**: Modellen krijgen de afbeelding en een eenvoudige vraag: "Is deze locatie verlaten?"
3. **Parsing**: De respons van het model wordt geparsed op '0' (niet verlaten) of '1' (verlaten)
4. **Resultaten**: Voorspellingen worden opgeslagen met de juiste labels voor het berekenen van nauwkeurigheid