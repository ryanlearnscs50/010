"""
Step 5: make the journal-submission version of the manuscript (BMC style).
Figures are removed from the text (they are uploaded as separate files) and their legends
are listed in a 'Figure legends' section after the references. Additional file 1 is split out.

Run from the 5_resubmission folder:  python -I code/05_make_submission_version.py
"""
import re
from pathlib import Path

MS = Path(__file__).resolve().parents[1] / "manuscript"
src = (MS / "manuscript.tex").read_text(encoding="utf8")

# split off Additional file 1
main, add = src.split(r"\section*{Additional file 1}")
add = add.replace(r"\end{document}", "")

# collect and remove figure environments from the main text, in order of appearance
figs = re.findall(r"\\begin\{figure\}\[H\](.*?)\\end\{figure\}", main, flags=re.S)
legends = []
for k, body in enumerate(figs, 1):
    cap = re.search(r"\\caption\{(.*)\}\s*\\label", body, flags=re.S).group(1)
    legends.append(f"\\paragraph{{Figure {k}.}} {cap}\n")
main = re.sub(r"\\begin\{figure\}\[H\].*?\\end\{figure\}\n*", "", main, flags=re.S)

legend_block = "\\clearpage\n\\section*{Figure legends}\n" + "\n".join(legends) + "\n"
main = main.replace(r"\clearpage" + "\n" + r"\section*{Tables}", legend_block + r"\clearpage" + "\n" + r"\section*{Tables}")
main = main.replace(r"\section*{Tables}", r"\section*{Tables}")
main += "\n" + r"\section*{Additional files}" + "\n" + \
        r"\paragraph{Additional file 1} (PDF). Table S1: sensitivity analyses. Figure S1: TreeSHAP feature attributions." + \
        "\n\\end{document}\n"
# figure references in text now point to legend numbers via labels -> replace \ref{fig:x} with numbers
labels = re.findall(r"\\label\{(fig:[^}]+)\}", "".join(figs))
for k, lab in enumerate(labels, 1):
    main = main.replace(f"\\ref{{{lab}}}", str(k))
(MS / "manuscript_submission.tex").write_text(main, encoding="utf8")

# Additional file 1 as its own document
preamble = src.split(r"\begin{document}")[0].replace(r"\linenumbers", "")
add_doc = preamble + r"\begin{document}" + "\n" + r"\section*{Additional file 1}" + \
          "\nNeo RYS. Machine-learning completion of sparse audiograms for hearing-loss triage: external validation and subgroup audit in NHANES adults.\n" + \
          add + "\n\\end{document}\n"
(MS / "additional_file_1.tex").write_text(add_doc, encoding="utf8")
print(f"{len(figs)} figures moved to legends; wrote manuscript_submission.tex and additional_file_1.tex")
