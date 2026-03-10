from typing import List, Tuple
from functools import lru_cache

from transformers import PreTrainedTokenizerFast

from .concept import Concept, ConceptCategory, TokenConceptFunctionTagger

try:
    from Bio.Align import substitution_matrices
except ImportError:
    substitution_matrices = None


@lru_cache(maxsize=1)
def _get_blosum62_matrix():
    """Cached function to load BLOSUM62 matrix."""
    if substitution_matrices is None:
        raise ImportError("Bio.Align.substitution_matrices is required. Install with: pip install biopython")
    return substitution_matrices.load("BLOSUM62")


@lru_cache(maxsize=1)
def _get_blosum_cliques(threshold: int = 0) -> Tuple[Tuple[str, ...], ...]:
    try:
        import networkx as nx
        from itertools import combinations
    except ImportError:
        raise ImportError("networkx is required for BLOSUM clique computation. Install with: pip install networkx")
    
    STANDARD_AA = [
        "A","R","N","D","C","Q","E","G","H","I",
        "L","K","M","F","P","S","T","W","Y","V"
    ]
    
    blosum = _get_blosum62_matrix()
    
    G = nx.Graph()
    G.add_nodes_from(STANDARD_AA)
    
    for i, a in enumerate(STANDARD_AA):
        for j, b in enumerate(STANDARD_AA):
            if i < j and blosum[a, b] >= threshold:
                G.add_edge(a, b)
    
    all_c = set()
    for mc in nx.find_cliques(G):
        mc = tuple(sorted(mc))
        for k in range(2, len(mc) + 1):
            for sub in combinations(mc, k):
                all_c.add(sub)
    
    return tuple(sorted(all_c, key=lambda c: (len(c), c)))


def _amino_acids_to_token_ids(amino_acids: List[str], tokenizer: PreTrainedTokenizerFast) -> List[int]:
    token_ids = []
    for aa in amino_acids:
        try:
            token_id = tokenizer.convert_tokens_to_ids(aa)
            if token_id == tokenizer.unk_token_id:
                raise ValueError(f"Amino acid '{aa}' maps to UNK token")
            token_ids.append(token_id)
        except Exception as e:
            raise ValueError(f"Could not convert amino acid '{aa}' to token ID: {e}")
    
    if len(token_ids) == 0:
        raise ValueError(f"No valid amino acid token IDs found for: {amino_acids}")
    
    return token_ids


def make_amino_acid_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    STANDARD_AA_LIST = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    ]
    ALL_AMINO_ACIDS = STANDARD_AA_LIST
    
    concepts = []
    
    for aa in ALL_AMINO_ACIDS:
        try:
            token_id = tokenizer.convert_tokens_to_ids(aa)
            if token_id == tokenizer.unk_token_id:
                print(f"  ⚠ Warning: Amino acid '{aa}' maps to UNK token, skipping concept creation")
                continue
        except Exception as e:
            print(f"  ⚠ Warning: Could not convert amino acid '{aa}' to token ID: {e}, skipping")
            continue
        
        concept_name = f"{aa}_amino_acid"
        concepts.append(Concept(
            concept_name=concept_name,
            concept_category=ConceptCategory.BIOLOGY,
            concept_function_tagger=TokenConceptFunctionTagger(
                token_values=[token_id],
                concept_name=concept_name,
                to_exclude_eos=False,
                to_exclude_bos=False,
                to_exclude_mask=True
            )
        ))
    
    return concepts


def make_polarity_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    POLAR_AA_IMGT = ['R', 'N', 'D', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y']
    
    concepts = []
    polar_token_ids = _amino_acids_to_token_ids(POLAR_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="polar_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=polar_token_ids,
            concept_name="polar_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    NON_POLAR_AA_IMGT = ['A', 'C', 'G', 'I', 'L', 'M', 'F', 'P', 'V', 'W']
    non_polar_token_ids = _amino_acids_to_token_ids(NON_POLAR_AA_IMGT, tokenizer)
    
    concepts.append(Concept(
        concept_name="non_polar_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=non_polar_token_ids,
            concept_name="non_polar_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,
        )
    ))
    
    return concepts


def make_hydropathy_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    HYDROPHOBIC_AA_IMGT = ['A', 'C', 'I', 'L', 'M', 'F', 'W', 'V']
    NEUTRAL_AA_IMGT = ['G', 'H', 'P', 'S', 'T', 'Y']
    HYDROPHILIC_AA_IMGT = ['R', 'N', 'D', 'Q', 'E', 'K']
    
    concepts = []
    hydrophobic_token_ids = _amino_acids_to_token_ids(HYDROPHOBIC_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="hydrophobic_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=hydrophobic_token_ids,
            concept_name="hydrophobic_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    neutral_token_ids = _amino_acids_to_token_ids(NEUTRAL_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="neutral_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=neutral_token_ids,
            concept_name="neutral_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    hydrophilic_token_ids = _amino_acids_to_token_ids(HYDROPHILIC_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="hydrophilic_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=hydrophilic_token_ids,
            concept_name="hydrophilic_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    return concepts


def make_volume_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    VERY_SMALL_AA_IMGT = ['A', 'G', 'S']
    SMALL_AA_IMGT = ['N', 'D', 'C', 'P', 'T']
    MEDIUM_AA_IMGT = ['Q', 'E', 'H', 'V']
    LARGE_AA_IMGT = ['R', 'I', 'L', 'K', 'M']
    VERY_LARGE_AA_IMGT = ['F', 'W', 'Y']
    
    concepts = []
    
    very_small_token_ids = _amino_acids_to_token_ids(VERY_SMALL_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="very_small_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=very_small_token_ids,
            concept_name="very_small_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    small_token_ids = _amino_acids_to_token_ids(SMALL_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="small_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=small_token_ids,
            concept_name="small_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    medium_token_ids = _amino_acids_to_token_ids(MEDIUM_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="medium_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=medium_token_ids,
            concept_name="medium_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    large_token_ids = _amino_acids_to_token_ids(LARGE_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="large_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=large_token_ids,
            concept_name="large_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    very_large_token_ids = _amino_acids_to_token_ids(VERY_LARGE_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="very_large_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=very_large_token_ids,
            concept_name="very_large_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    return concepts


def make_chemical_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    ALIPHATIC_AA_IMGT = ['A', 'G', 'I', 'L', 'P', 'V']
    ALIPHATIC_IMGT_11_AA = ['A', 'I', 'L', 'V']
    AROMATIC_AA_IMGT = ['F', 'W', 'Y']
    AROMATIC_RING_AA = ['F', 'Y', 'W', 'H']
    SULFUR_AA_IMGT = ['C', 'M']
    HYDROXYL_AA_IMGT = ['S', 'T']
    BASIC_AA_IMGT = ['R', 'H', 'K']
    ACIDIC_AA_IMGT = ['D', 'E']
    AMIDE_AA_IMGT = ['N', 'Q']
    
    concepts = []
    
    aliphatic_token_ids = _amino_acids_to_token_ids(ALIPHATIC_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="aliphatic_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=aliphatic_token_ids,
            concept_name="aliphatic_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    aliphatic_imgt_11_token_ids = _amino_acids_to_token_ids(ALIPHATIC_IMGT_11_AA, tokenizer)
    concepts.append(Concept(
        concept_name="aliphatic_amino_acid_IMGT_11",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=aliphatic_imgt_11_token_ids,
            concept_name="aliphatic_amino_acid_IMGT_11",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    aromatic_token_ids = _amino_acids_to_token_ids(AROMATIC_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="aromatic_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=aromatic_token_ids,
            concept_name="aromatic_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    aromatic_ring_token_ids = _amino_acids_to_token_ids(AROMATIC_RING_AA, tokenizer)
    concepts.append(Concept(
        concept_name="aromatic_ring_amino_acid",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=aromatic_ring_token_ids,
            concept_name="aromatic_ring_amino_acid",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    sulfur_token_ids = _amino_acids_to_token_ids(SULFUR_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="sulfur_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=sulfur_token_ids,
            concept_name="sulfur_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    hydroxyl_token_ids = _amino_acids_to_token_ids(HYDROXYL_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="hydroxyl_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=hydroxyl_token_ids,
            concept_name="hydroxyl_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    basic_token_ids = _amino_acids_to_token_ids(BASIC_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="basic_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=basic_token_ids,
            concept_name="basic_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    acidic_token_ids = _amino_acids_to_token_ids(ACIDIC_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="acidic_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=acidic_token_ids,
            concept_name="acidic_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    amide_token_ids = _amino_acids_to_token_ids(AMIDE_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="amide_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=amide_token_ids,
            concept_name="amide_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    return concepts


def make_charge_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    POSITIVE_CHARGED_AA_IMGT = ['R', 'H', 'K']
    NEGATIVE_CHARGED_AA_IMGT = ['D', 'E']
    POLAR_UNCHARGED_AA_IMGT = ['S', 'T', 'N', 'Q', 'Y']
    
    concepts = []
    
    positive_charged_token_ids = _amino_acids_to_token_ids(POSITIVE_CHARGED_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="positive_charged_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=positive_charged_token_ids,
            concept_name="positive_charged_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    negative_charged_token_ids = _amino_acids_to_token_ids(NEGATIVE_CHARGED_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="negative_charged_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=negative_charged_token_ids,
            concept_name="negative_charged_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    polar_uncharged_token_ids = _amino_acids_to_token_ids(POLAR_UNCHARGED_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="polar_uncharged_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=polar_uncharged_token_ids,
            concept_name="polar_uncharged_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    return concepts


def make_hydrogen_donor_acceptor_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    DONOR_AA_IMGT = ['R', 'K', 'W']
    ACCEPTOR_AA_IMGT = ['D', 'E']
    DONOR_AND_ACCEPTOR_AA_IMGT = ['N', 'Q', 'H', 'S', 'T', 'Y']
    NONE_HYDROGEN_DONOR_ACCEPTOR_AA_IMGT = ['A', 'C', 'G', 'I', 'L', 'M', 'F', 'P', 'V']
    
    concepts = []
    
    donor_token_ids = _amino_acids_to_token_ids(DONOR_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="hydrogen_donor_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=donor_token_ids,
            concept_name="hydrogen_donor_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    acceptor_token_ids = _amino_acids_to_token_ids(ACCEPTOR_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="hydrogen_acceptor_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=acceptor_token_ids,
            concept_name="hydrogen_acceptor_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    donor_and_acceptor_token_ids = _amino_acids_to_token_ids(DONOR_AND_ACCEPTOR_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="hydrogen_donor_and_acceptor_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=donor_and_acceptor_token_ids,
            concept_name="hydrogen_donor_and_acceptor_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    none_hydrogen_donor_acceptor_token_ids = _amino_acids_to_token_ids(NONE_HYDROGEN_DONOR_ACCEPTOR_AA_IMGT, tokenizer)
    concepts.append(Concept(
        concept_name="no_hydrogen_donor_acceptor_amino_acid_IMGT",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=none_hydrogen_donor_acceptor_token_ids,
            concept_name="no_hydrogen_donor_acceptor_amino_acid_IMGT",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    return concepts


def make_secondary_structure_propensity_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    ALPHA_HELIX_FORMERS_AA = ['A', 'E', 'L', 'M', 'Q', 'K', 'R', 'H']
    BETA_SHEET_FORMERS_AA = ['V', 'I', 'Y', 'F', 'T', 'C', 'W']
    TURN_FORMERS_AA = ['N', 'D', 'S']
    P_G_AA = ['P', 'G']
    
    concepts = []
    
    alpha_helix_formers_token_ids = _amino_acids_to_token_ids(ALPHA_HELIX_FORMERS_AA, tokenizer)
    concepts.append(Concept(
        concept_name="alpha_helix_former_amino_acid_propensity",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=alpha_helix_formers_token_ids,
            concept_name="alpha_helix_former_amino_acid_propensity",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    beta_sheet_formers_token_ids = _amino_acids_to_token_ids(BETA_SHEET_FORMERS_AA, tokenizer)
    concepts.append(Concept(
        concept_name="beta_sheet_former_amino_acid_propensity",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=beta_sheet_formers_token_ids,
            concept_name="beta_sheet_former_amino_acid_propensity",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    turn_formers_token_ids = _amino_acids_to_token_ids(TURN_FORMERS_AA, tokenizer)
    concepts.append(Concept(
        concept_name="turn_former_amino_acid_propensity",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=turn_formers_token_ids,
            concept_name="turn_former_amino_acid_propensity",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    p_g_token_ids = _amino_acids_to_token_ids(P_G_AA, tokenizer)
    concepts.append(Concept(
        concept_name="helix_breakers_amino_acid_propensity",
        concept_category=ConceptCategory.BIOLOGY,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=p_g_token_ids,
            concept_name="helix_breakers_amino_acid_propensity",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=True,

        )
    ))
    
    return concepts


def make_blosum_cluster_concepts(tokenizer: PreTrainedTokenizerFast, threshold: int = 0) -> list:
    cliques = _get_blosum_cliques(threshold=threshold)
    
    concepts = []
    
    for clique in cliques:
        clique_list = list(clique)
        cluster_name = "".join(sorted(clique_list))
        concept_name = f"blosum_cluster_{cluster_name}"
        
        cluster_token_ids = _amino_acids_to_token_ids(clique_list, tokenizer)
        concepts.append(Concept(
            concept_name=concept_name,
            concept_category=ConceptCategory.BIOLOGY,
            concept_function_tagger=TokenConceptFunctionTagger(
                token_values=cluster_token_ids,
                concept_name=concept_name,
                to_exclude_eos=False,
                to_exclude_bos=False,
                to_exclude_mask=True,
            )
        ))
    
    return concepts


def make_biochemical_properties_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    concepts = []
    concepts.extend(make_polarity_concepts(tokenizer))
    concepts.extend(make_hydropathy_concepts(tokenizer))
    concepts.extend(make_volume_concepts(tokenizer))
    concepts.extend(make_chemical_concepts(tokenizer))
    concepts.extend(make_charge_concepts(tokenizer))
    concepts.extend(make_hydrogen_donor_acceptor_concepts(tokenizer))
    concepts.extend(make_secondary_structure_propensity_concepts(tokenizer))
    concepts.extend(make_blosum_cluster_concepts(tokenizer, threshold=0))
    amino_acid_concepts = make_amino_acid_concepts(tokenizer)
    concepts.extend(amino_acid_concepts)
    return concepts

