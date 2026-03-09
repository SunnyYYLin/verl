import numpy as np

from alphagenome_research.model import dna_model
from alphagenome_research.model import dna_output
from alphagenome_research.model.metadata import metadata

from alphagenome.data import ontology


# =========================
# sequence layout
# =========================

SEQ_LEN = 8192

ENH_CENTER = 2048
TSS_CENTER = 6144

ENH_SIZE = 512
TSS_SIZE = 2048

ENH_START = ENH_CENTER - ENH_SIZE // 2
ENH_END = ENH_CENTER + ENH_SIZE // 2

TSS_START = TSS_CENTER - TSS_SIZE // 2
TSS_END = TSS_CENTER + TSS_SIZE // 2


# =========================
# requested tracks
# =========================

REQUESTED_OUTPUTS = [
    dna_output.OutputType.ATAC,
    dna_output.OutputType.DNASE,
    dna_output.OutputType.CHIP_HISTONE,
    dna_output.OutputType.CHIP_TF,
    dna_output.OutputType.PROCAP,
    dna_output.OutputType.CAGE,
    dna_output.OutputType.RNA_SEQ
]


# =========================
# track ontology mapping
# =========================

TRACK_ONTOLOGY = {

    "ATAC": ["EFO:0002067"],
    "DNASE": ["EFO:0002067"],
    "CHIP_HISTONE": ["EFO:0002067"],
    "CHIP_TF": ["EFO:0002067"],
    "PROCAP": ["ENCSR740IPL"],

    "CAGE": ["EFO:0002067"],
    "RNA_SEQ": ["EFO:0002067"]
}


# =========================
# helper
# =========================

def build_ontology_terms(curie_list):

    return [
        ontology.OntologyTerm(ontology_curie=c)
        for c in curie_list
    ]


def extract_track_values(output, output_type):

    pred = getattr(output, output_type.lower()).values

    if len(pred.shape) == 2:
        pred = np.mean(pred, axis=0)

    return pred


# =========================
# compute region scores
# =========================

def compute_scores(output, track_masks):

    results = {}

    for output_type in REQUESTED_OUTPUTS:

        name = output_type.name

        if output_type not in track_masks:
            continue

        mask = track_masks[output_type]

        pred = getattr(output, output_type.lower()).values

        if len(pred.shape) == 2:

            pred = pred[mask]

            if pred.shape[0] == 0:
                continue

            pred = np.mean(pred, axis=0)

        if name in ["CAGE", "RNA_SEQ"]:

            region = pred[TSS_START:TSS_END]

            results[f"{name}_TSS"] = float(np.mean(region))

        else:

            region = pred[ENH_START:ENH_END]

            results[f"{name}_enhancer"] = float(np.mean(region))

    return results


# =========================
# delta
# =========================

def compute_delta(gen_scores, orig_scores):

    delta = {}

    for k in gen_scores:

        if k in orig_scores:
            delta[k] = gen_scores[k] - orig_scores[k]

    return delta


# =========================
# build track masks
# =========================

def build_track_masks(output_metadata):

    track_masks = {}

    for output_type in REQUESTED_OUTPUTS:

        name = output_type.name

        if name not in TRACK_ONTOLOGY:
            continue

        ontologies = build_ontology_terms(TRACK_ONTOLOGY[name])

        masks = metadata.create_track_masks(
            metadata=output_metadata,
            requested_outputs=[output_type],
            requested_ontologies=ontologies
        )

        if output_type in masks:
            track_masks[output_type] = masks[output_type]

    return track_masks


# =========================
# prediction
# =========================

def predict(model, sequence, organism):

    return model.predict_sequence(
        sequence=sequence,
        organism=organism,
        requested_outputs=REQUESTED_OUTPUTS,
        interval=None
    )


# =========================
# pipeline
# =========================

def run_pipeline(
    model_path,
    original_sequence,
    generated_sequence,
    organism=dna_model.Organism.HOMO_SAPIENS
):

    print("Loading AlphaGenome model...")
    model = dna_model.create(model_path)

    print("Loading metadata...")
    output_metadata = metadata.load(organism)

    print("Building track masks...")
    track_masks = build_track_masks(output_metadata)

    print("Predicting original sequence...")
    orig_output = predict(model, original_sequence, organism)

    print("Predicting generated sequence...")
    gen_output = predict(model, generated_sequence, organism)

    print("Computing scores...")

    orig_scores = compute_scores(orig_output, track_masks)
    gen_scores = compute_scores(gen_output, track_masks)

    delta_scores = compute_delta(gen_scores, orig_scores)

    return {
        "original": orig_scores,
        "generated": gen_scores,
        "delta": delta_scores
    }


# =========================
# example run
# =========================

if __name__ == "__main__":

    MODEL_PATH = "/vepfs-mlp2/mlp-public/zhongcuiting/models/alphagenome-all-folds"

    ORIGINAL_SEQUENCE = "ATCG" * 2048
    GENERATED_SEQUENCE = "ATCG" * 2048

    results = run_pipeline(
        MODEL_PATH,
        ORIGINAL_SEQUENCE,
        GENERATED_SEQUENCE
    )

    print("\n===== RESULTS =====")

    for group, values in results.items():

        print(f"\n{group}")

        for k, v in values.items():
            print(f"{k}: {v:.6f}")