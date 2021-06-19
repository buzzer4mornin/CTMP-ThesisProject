import sys
sys.path.insert(0, '../../common')
import perplexity_vb

def compute_perplexities_vb(beta, alpha, eta, max_iter, corpusids_part1,
                            wordids_1, wordcts_1, wordids_2, wordcts_2):
    vb = per_vb.VB(beta, alpha, eta, max_iter)
    LD2 = vb.compute_perplexity(corpusids_part1[k], corpuscts_part1[k], corpusids_part2[k], corpuscts_part2[k])
    return(LD2)

LD2 = compute_perplexities_vb(ml_ope.beta, ddict['alpha'], ddict['eta'], ddict['iter_infer'],\
                                                                wordids_1, wordcts_1, wordids_2, wordcts_2)
