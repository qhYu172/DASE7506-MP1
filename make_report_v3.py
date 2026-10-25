"""Build the final three-page report, including attribution and development history."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

root=Path(__file__).resolve().parent
out=root/'output/pdf/MP2_report.pdf';out.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold','C:/Windows/Fonts/arialbd.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='T',fontName='ArialBold',fontSize=17,leading=21,spaceAfter=8,textColor=colors.HexColor('#18344f')))
styles.add(ParagraphStyle(name='H',fontName='ArialBold',fontSize=11,leading=14,spaceBefore=12,spaceAfter=6,textColor=colors.HexColor('#176088')))
styles.add(ParagraphStyle(name='B',fontName='Arial',fontSize=9.2,leading=13,spaceAfter=7))
styles.add(ParagraphStyle(name='S',fontName='Arial',fontSize=8,leading=11,spaceAfter=5))
story=[]
def p(text,style='B'):story.append(Paragraph(text,styles[style]))
def h(text):p(text,'H')
def table(rows,widths):
    t=Table([[Paragraph(str(v),styles['S']) for v in row] for row in rows],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f2f7')),
      ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7fafc')]),
      ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(t);story.append(Spacer(1,7))

p('Selecting Reliable LLM Reasoning','T')
p('DASE7506 Mini-project 2 | Final method v3 | 25 October 2026','S')
p('Yu Qinghao | Student ID: 3036843599','B')
h('1. Research question')
p('Does comparing distinct numeric answers improve selection over scoring each reasoning trace independently? The hypothesis is that pooling same-answer candidates reduces redundant evidence, while question-weighted ranking focuses training on the actual choice among competing answers. A second hypothesis is that inexpensive numeric plausibility features add information beyond the fixed embeddings.')
h('2. Data, protocol and development history')
p('The supplied bank has 4,755 train, 798 validation and 467 test questions, four candidates per question, and a 1,024-dimensional embedding per candidate. All fitting uses train inputs and references. Model comparisons use train-question cross-validation and validation. Test grading uses the unchanged official final-answer parser and scorer.')
p('The train set contains 2,114 questions with at least two distinct valid parsed answers; validation contains 355. These groups drive fitting and development comparisons. Validation also includes 379 questions with four identical parsed answers, which makes whole-set validation accuracy substantially easier than test. The test split is entirely made of conflicting-answer groups.')
p('V1 and v2 aggregate test scores were known before this follow-up. Public authors\' README scores were also seen during the literature/code survey. V3 selection used train/validation, and its source/asset hashes were frozen before its test run. No per-question test errors, test-derived features, answer lookup tables or other authors\' checkpoints were used. This history is disclosed; the study is not claimed to be a pristine blind-test experiment.')
h('3. Final method: pooled answer groups with pairwise subset training')
p('Candidates are grouped by the exact numeric value returned by the official parser. Each answer group receives 1,112 features: a mean embedding (1,024); four support/dispersion descriptors; means and ranges of 39 existing text/similarity features (78); and six numeric descriptors. The latter describe integer status, magnitude relative to question numbers, denominator size, matching a question number, number count and nearest-number distance. They are plausibility cues, not an arithmetic proof.')
p('Training enumerates nonempty subsets of each same-answer group. For each question it compares representations of correct-answer subsets with wrong-answer subsets. Pair weights sum to one per question; thus questions with more combinations do not dominate. Normalization is fitted on full training groups inside each fold, before augmentation. A linear score is fitted with weighted logistic pair loss plus L2 regularization. The selected coefficient is 0.01. Each final answer group is scored once at inference; any member of the highest-scoring group has the same graded answer.')
p('This implementation adapts general ideas from public repositories [1,2], combining them with this project\'s surface features and numeric descriptors. No external source file or checkpoint was copied. The original v2 and the new method were also compared with an equal probability fusion; the standalone answer-group model won on validation. Source attribution is expanded on page 3.')

story.append(PageBreak())
h('4. Controlled comparisons')
p('All CV rows below use the same three shuffled folds of the 2,114 conflicting-answer train questions, seed 7506. Every fold refits normalization and coefficients using only its training questions. The table reports each family\'s best tried regularization; these are selection estimates, not nested-CV estimates.')
table([['Variant','Train CV accuracy'],
 ['Group softmax, embedding/support only','80.13%'],
 ['Group softmax, pooled text features','80.75%'],
 ['Pairwise subsets, embedding/support only','81.13%'],
 ['Pairwise subsets, pooled text features','81.74%'],
 ['Pairwise subsets, pooled + numeric features','82.31% (1,740/2,114)']], [119*mm,47*mm])
p('The pairwise search checked alpha in {0.001, 0.01, 0.1} for three feature sets. The earlier softmax search used {0.001, 0.01, 0.1, 1.0}. In the matched pairwise comparison, numeric descriptors add 12 CV successes over pooled text alone. The softmax versus pairwise-subset comparison changes both objective and augmentation, so it does not isolate the contribution of augmentation by itself.')
table([['Validation comparison','All 798','Hard 355'],
 ['Previous v2','740 (92.73%)','297 (83.66%)'],
 ['Group softmax, pooled','741 (92.86%)','298 (83.94%)'],
 ['Pairwise subsets, pooled','740 (92.73%)','297 (83.66%)'],
 ['Pairwise subsets, pooled + numeric (v3)','746 (93.48%)','303 (85.35%)'],
 ['Equal fusion of v2 and v3','742 (92.98%)','299 (84.23%)']], [94*mm,36*mm,36*mm])
p('The final v3 beats v2 by six validation questions and the equal fusion by four. This supports the selected adaptation on these data, but the small margins and repeated development comparisons limit confidence in generalization. Fusion is a negative result: adding the older scorer did not help.')
h('5. Frozen official result and resource use')
table([['Measure','Recorded result'],
 ['V1 complete test','366/467 = 78.37%'],
 ['V2 complete test','372/467 = 79.66%'],
 ['Final v3 complete test','390/467 = 83.5117773019272%'],
 ['V3 fresh prediction + independent scorer','1.265 seconds, four threads'],
 ['V3 additional inference assets','28,205 bytes (27.54 KiB)'],
 ['Local environment','Intel Core i9-14900HX, Windows 11, Python 3.12.10']], [88*mm,78*mm])
p('The net test gain over v2 is 18 correct questions (+3.85 percentage points). Prediction and scoring run in fresh serial processes, including startup, imports, loading and preprocessing. The measured time is local; eligibility is determined on the reference Xeon Platinum 8457C. The final nine-setting CV search plus refit took 53.66 s locally. Earlier recorded group searches took 18.29 s and 41.18 s; these costs exclude prior v1/v2 exploration.')

story.append(PageBreak())
h('6. Limitations and interpretation')
p('Grouping suppresses redundant traces, but agreement can still reflect a shared mistake. Surface and numeric descriptors can favor plausible-looking wrong answers; they do not verify that every condition in the problem was used. The scorer itself grades only the final numeric answer, so these results should not be described as a measure of reasoning validity. The current method remains below 89% on test and is not claimed optimal.')
p('The v1 score-margin diagnostic was nearly flat across hard-validation quartiles (79.55%, 80.90%, 78.65%, 80.90%). V3 probability calibration has not been established, and its scores should not be interpreted as trustworthy correctness probabilities. Generalization to an unseen instructor audit bank has not been measured.')
h('7. Reproduction and integrity')
p('Use Python 3.12 and the pinned packages in code/requirements.txt: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0. From code/, run the commands below. The evaluation wrapper verifies the frozen source/asset hashes, sets the four-thread environment, and invokes the unchanged official prediction and scoring scripts.')
for line in ['py -3.12 -m pip install -r requirements.txt','py -3.12 check_data.py',
             'py -3.12 evaluate_frozen.py --split validation','py -3.12 evaluate_frozen.py --split test']:
    p(line,'S')
p('No retraining is required. The checkpoint is code/runs/model_bundle_v3/. To reproduce training and the CV grid, run train_augmented_ranker.py; it writes a separate runs/augmented_ranker/ result. Train and validation labels are used for their stated purposes; the training program never opens test. See RESULTS.md for direct official commands and the earlier ablations.')
p('The exact code/asset identity is recorded in code/runs/v3-freeze.json. Official test result and resource records are test-v3-score.json and test-v3-resources.json. SHA-256 of ranker.npz: 5c5c0f4a560c1f42acfb27c82abe9dbaacb75090dc8c9c18cf5be9ef33285723. Contract checks on validation inputs passed for empty input, output types, unseen IDs, candidate permutation, batch independence and unchanged v2 predictions after refactoring.')
h('8. Attribution, reuse and AI assistance')
p('The submission-site artifact URLs were sealed at the time of inspection. The following independently public repositories were found through GitHub search. Their README scores are author claims, not reproductions conducted here. Public methods informed design choices; all final coefficients were trained locally from the supplied train split. No external dataset or additional pretrained weights were used.')
refs=[
 '[1] CaspianW/dase7506-mp2: answer-group pooling, equal question weighting and within-answer subset augmentation. https://github.com/CaspianW/dase7506-mp2',
 '[2] vectorBH6/DASE7506-MP2: group-level supervision and fold-local normalization. https://github.com/vectorBH6/DASE7506-MP2',
 '[3] Lyan9145/DASE7506-MP2: DeepSets/listwise ideas considered; its neural architecture is not included. https://github.com/Lyan9145/DASE7506-MP2',
 '[4] CARDINAL-02/DASE7506-MP2-Submission: arithmetic and coverage features inspected; implementation not included. https://github.com/CARDINAL-02/DASE7506-MP2-Submission']
for ref in refs:p(ref,'S')
p('PUBLIC_METHODS.md records inspected tree identities and which ideas were adapted. The inspected source trees did not expose a repository-wide code license; no source file or checkpoint was copied. OpenAI Codex substantially assisted with research, independent implementation, experiments, debugging and report drafting. The supplied data/evaluator are reused unchanged. GSM8K and course Qwen3-0.6B provenance and license notices remain in code/data/.','S')

def footer(c,d):
    c.saveState();c.setStrokeColor(colors.HexColor('#c6d4de'));c.line(20*mm,18*mm,190*mm,18*mm)
    c.setFont('Arial',7);c.setFillColor(colors.HexColor('#617487'))
    c.drawString(20*mm,13*mm,'DASE7506 MP2 | Final method v3');c.drawRightString(190*mm,13*mm,f'Page {d.page}');c.restoreState()
SimpleDocTemplate(str(out),pagesize=A4,leftMargin=21*mm,rightMargin=21*mm,
                  topMargin=18*mm,bottomMargin=24*mm).build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
