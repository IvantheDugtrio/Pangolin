import argparse

import gffutils
import gzip
import pandas as pd
import pyfastx
import pysam
from pkg_resources import resource_filename

from pangolin.model import *


IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def one_hot_encode(seq, strand):
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        seq = np.asarray(list(map(int, list(seq))))
    elif strand == '-':
        seq = np.asarray(list(map(int, list(seq[::-1]))))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype('int8')]


def compute_score(ref_seq, alt_seq, strand, d, models, compute_platform):
    ref_seq = one_hot_encode(ref_seq, strand).T
    ref_seq = torch.from_numpy(np.expand_dims(ref_seq, axis=0)).float()
    alt_seq = one_hot_encode(alt_seq, strand).T
    alt_seq = torch.from_numpy(np.expand_dims(alt_seq, axis=0)).float()

    ref_seq = ref_seq.to(torch.device(compute_platform))
    alt_seq = alt_seq.to(torch.device(compute_platform))

    pangolin = list()
    for j in range(4):
        score = list()
        for model in models[3*j:3*j+3]:
            with torch.no_grad():
                ref = model(ref_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                alt = model(alt_seq)[0][[1,4,7,10][j],:].cpu().numpy()
                if strand == '-':
                    ref = ref[::-1]
                    alt = alt[::-1]
                l = 2*d+1
                ndiff = np.abs(len(ref)-len(alt))
                if len(ref)>len(alt):
                    alt = np.concatenate([alt[0:l//2+1],np.zeros(ndiff),alt[l//2+1:]])
                elif len(ref)<len(alt):
                    alt = np.concatenate([alt[0:l//2],np.max(alt[l//2:l//2+ndiff+1], keepdims=True),alt[l//2+ndiff+1:]])
                score.append(alt-ref)
        pangolin.append(np.mean(score, axis=0))

    pangolin = np.array(pangolin)
    loss = pangolin[np.argmin(pangolin, axis=0), np.arange(pangolin.shape[1])]
    gain = pangolin[np.argmax(pangolin, axis=0), np.arange(pangolin.shape[1])]
    return loss, gain


def get_genes(chr, pos, gtf):
    genes = gtf.region((chr, pos-1, pos-1), featuretype="gene")
    genes_pos, genes_neg = {}, {}

    for gene in genes:
        if gene[3] > pos or gene[4] < pos:
            continue
        gene_id = gene["gene_id"][0]
        exons = []
        for exon in gtf.children(gene, featuretype="exon"):
            exons.extend([exon[3], exon[4]])
        if gene[6] == '+':
            genes_pos[gene_id] = exons
        elif gene[6] == '-':
            genes_neg[gene_id] = exons

    return (genes_pos, genes_neg)


def process_variant(lnum, chr, pos, ref, alt, gtf, models, compute_platform, args):
    warning_msg_base = f"[{chr} {pos} {ref} {alt}] Warning, skipping variant: "
    d = args.distance
    cutoff = args.score_cutoff

    if len(set("ACGT").intersection(set(ref))) == 0 or len(set("ACGT").intersection(set(alt))) == 0 \
            or (len(ref) != 1 and len(alt) != 1 and len(ref) != len(alt)):
        print(warning_msg_base + "Variant format not supported.")
        return -1
    elif len(ref) > 2*d:
        print(warning_msg_base + "skipping variant: Deletion too large")
        return -1

    fasta = pyfastx.Fasta(args.reference_file)
    # try to make vcf chromosomes compatible with reference chromosomes
    if chr not in fasta.keys() and "chr"+chr in fasta.keys():
        chr = "chr"+chr
    elif chr not in fasta.keys() and chr[3:] in fasta.keys():
        chr = chr[3:]

    try:
        seq = fasta[chr][pos-5001-d:pos+len(ref)+4999+d].seq
    except Exception as e:
        print(e)
        print(warning_msg_base + "Could not get sequence, possibly because the variant is too close to chromosome ends.\nSee error message above.")
        return -1

    if seq[5000+d:5000+d+len(ref)] != ref:
        print(warning_msg_base + "Mismatch between FASTA (ref base: %s) and variant file (ref base: %s)."
              % (seq[5000+d:5000+d+len(ref)], ref))
        return -1

    ref_seq = seq
    alt_seq = seq[:5000+d] + alt + seq[5000+d+len(ref):]

    # get genes that intersect variant
    genes_pos, genes_neg = get_genes(chr, pos, gtf)
    if len(genes_pos)+len(genes_neg)==0:
        print(warning_msg_base + "Variant not contained in a gene body. Do GTF/FASTA chromosome names match?")
        return -1

    # get splice scores
    loss_pos, gain_pos = None, None
    if len(genes_pos) > 0:
        loss_pos, gain_pos = compute_score(ref_seq, alt_seq, '+', d, models, compute_platform)
    loss_neg, gain_neg = None, None
    if len(genes_neg) > 0:
        loss_neg, gain_neg = compute_score(ref_seq, alt_seq, '-', d, models, compute_platform)

    scores_list = list()
    for (genes, loss, gain) in (
        (genes_pos,loss_pos,gain_pos),(genes_neg,loss_neg,gain_neg)
    ):
        # Emit a bundle of scores/warnings per gene; join them all later
        for gene, positions in genes.items():
            per_gene_scores = list()
            warnings = "Warnings:"
            positions = np.array(positions)
            positions = positions - (pos - d)

            if not args.no_mask and len(positions) != 0:
                positions_filt = positions[(positions>=0) & (positions<len(loss))]
                # set splice gain at annotated sites to 0
                gain[positions_filt] = np.minimum(gain[positions_filt], 0)
                # set splice loss at unannotated sites to 0
                not_positions = ~np.isin(np.arange(len(loss)), positions_filt)
                loss[not_positions] = np.maximum(loss[not_positions], 0)

            elif not args.no_mask:
                warnings += "NoAnnotatedSitesToMaskForThisGene"
                loss[:] = np.maximum(loss[:], 0)

            if args.score_exons:
                scores1 = [f"{gene}_sites1"]
                scores2 = [f"{gene}_sites2"]

                for i in range(len(positions)//2):
                    p1, p2 = positions[2*i], positions[2*i+1]
                    if p1<0 or p1>=len(loss):
                        s1 = "NA"
                    else:
                        s1 = [loss[p1],gain[p1]]
                        s1 = round(s1[np.argmax(np.abs(s1))],2)
                    if p2<0 or p2>=len(loss):
                        s2 = "NA"
                    else:
                        s2 = [loss[p2],gain[p2]]
                        s2 = round(s2[np.argmax(np.abs(s2))],2)
                    if s1 == "NA" and s2 == "NA":
                        continue
                    scores1.append(f"{p1-d}:{s1}")
                    scores2.append(f"{p2-d}:{s2}")
                per_gene_scores += scores1 + scores2

            elif cutoff:  # process if a cutoff is set
                per_gene_scores.append(gene)
                l, g = np.where(loss<=-cutoff)[0], np.where(gain>=cutoff)[0]
                for p, s in zip(np.concatenate([g-d,l-d]), np.concatenate([gain[g],loss[l]])):
                    per_gene_scores.append(f"{p}:{round(s,2)}")

            else:
                per_gene_scores.append(gene)
                l, g = np.argmin(loss), np.argmax(gain),
                gain_str = f"{g-d}:{round(gain[g],2)}"
                loss_str = f"{l-d}:{round(loss[l],2)}"
                per_gene_scores += [gain_str, loss_str]

            per_gene_scores.append(warnings)
            scores_list.append('|'.join(per_gene_scores))

    return ','.join(scores_list)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_file", help="VCF or CSV file with a header (see COLUMN_IDS option).")
    parser.add_argument("reference_file", help="FASTA file containing a reference genome sequence. Can be gzipped.")
    parser.add_argument("annotation_file", help="gffutils database file. Can be generated using create_db.py.")
    parser.add_argument("output_file", help="Prefix for output file. Will be a VCF/CSV if variant_file is VCF/CSV.")
    parser.add_argument("-c", "--column_ids", default="CHROM,POS,REF,ALT", help="(If variant_file is a CSV) Column IDs for: chromosome, variant position, reference bases, and alternative bases. "
                                                                                "Separate IDs by commas. (Default: CHROM,POS,REF,ALT)")
    parser.add_argument("--no_mask", action=argparse.BooleanOptionalAction, help="If present, splice gains (increases in score) at annotated splice sites and splice losses (decreases in score) at unannotated splice sites will not be set to 0. (Default: masked)")
    parser.add_argument("-s", "--score_cutoff", type=float, help="Output all sites with absolute predicted change in score >= cutoff, instead of only the maximum loss/gain sites.")
    parser.add_argument("-d", "--distance", type=int, default=50, help="Number of bases on either side of the variant for which splice scores should be calculated. (Default: 50)")
    parser.add_argument("--score_exons", action=argparse.BooleanOptionalAction, help="Output changes in score for both splice sites of annotated exons, as long as one splice site is within the considered range (specified by -d). Output will be: gene|site1_pos:score|site2_pos:score|...")
    return parser.parse_args()


def validate_gtf(gtf):
    try:
        gtf = gffutils.FeatureDB(gtf)
    except:
        print("ERROR, annotation_file could not be opened. Is it a gffutils database file?")
        exit(1)
    return gtf


def load_models():
    models = list()
    for i in [0,2,4,6]:
        for j in range(1,4):
            model = Pangolin(L, W, AR)
            if torch.cuda.is_available():
                compute_platform = 'cuda'
                model.cuda()
            elif torch.mps.is_available():
                compute_platform = 'mps'
                model.to('mps')
            elif torch.xpu.is_available():
                compute_platform = 'xpu'
                model.xpu()
            elif torch.rocm.is_available():
                compute_platform = 'rocm'
                model.rocm()
            else:
                compute_platform = 'cpu'
            print(f"Using {compute_platform} for model models/final.{j}.{i}.3.v2")
            weights = torch.load(resource_filename(__name__,f"models/final.{j}.{i}.3.v2"), map_location=torch.device(compute_platform))
            model.load_state_dict(weights)
            model.eval()
            models.append(model)
    return models, compute_platform


def annotate_variants(out_variant_file, variant_file, lnum, gtf, models, compute_platform, args):
    out_variant_file.header.add_meta(
        key="INFO",
        items=[
            ("ID", "Pangolin"),
            ("Number", "."),
            ("Type", "String"),
            (
                "Description",
                "Pangolin splice scores. Format: gene|pos:score_change|pos:score_change|warnings,..."
            ),
        ]
    )
    for i, variant_record in enumerate(variant_file):
        variant_record.translate(out_variant_file.header)
        assert variant_record.ref, f"Empty REF field in variant record {variant_record}"
        assert variant_record.alts, f"Empty ALT field in variant record {variant_record}"
        scores = process_variant(
            lnum + i,
            str(variant_record.contig),
            int(variant_record.pos),
            variant_record.ref,
            str(variant_record.alts[0]),
            gtf,
            models,
            compute_platform,
            args
        )
        if scores != -1:
            variant_record.info["Pangolin"] = scores
        out_variant_file.write(variant_record)


def count_hdr_lines(vcf_hdr):
    lnum = 0
    for line in vcf_hdr:
        lnum += 1
        if not line.startswith('#'):
            return lnum


def process_vcf(vcf_file, gtf, models, compute_platform, args):
    # count the number of header lines
    ext=''
    if vcf_file.endswith('.gz'):
        ext='.gz'
        with gzip.open(vcf_file, 'rt') as vcf_hdr:
            lnum = count_hdr_lines(vcf_hdr)
    else:
        with open(vcf_file, 'r') as vcf_hdr:
            lnum = count_hdr_lines(vcf_hdr)


    with pysam.VariantFile(vcf_file) as variant_file, pysam.VariantFile(
        f"{args.output_file}.vcf{ext}", 'w', header=variant_file.header
    ) as out_variant_file:
        annotate_variants(out_variant_file, variant_file, lnum, gtf, models, compute_platform, args)


def process_csv(csv_file, gtf, models, compute_platform, args):
    col_ids = args.column_ids.split(',')
    variants = pd.read_csv(csv_file, header=0)
    fout = open(f"{args.output_file}.csv", 'w')
    fout.write(','.join(variants.columns)+',Pangolin\n')
    fout.flush()

    for lnum, variant in variants.iterrows():
        chr, pos, ref, alt = variant[col_ids]
        ref, alt = ref.upper(), alt.upper()
        scores = process_variant(
            lnum + 1,
            str(chr),
            int(pos),
            ref,
            alt,
            gtf,
            models,
            compute_platform,
            args
        )
        if scores == -1:
            fout.write(','.join(variant.to_csv(header=False, index=False).split('\n'))+'\n')
        else:
            fout.write(','.join(variant.to_csv(header=False, index=False).split('\n'))+scores+'\n')
        fout.flush()

    fout.close()


def main():
    args = get_args()
    variants = args.variant_file
    gtf = validate_gtf(args.annotation_file)
    models, compute_platform = load_models()
    if 'vcf' in variants:
        process_vcf(variants, gtf, models, compute_platform, args)
    elif variants.endswith('.csv'):
        process_csv(variants, gtf, models, compute_platform, args)
    else:
        print("ERROR, variant_file needs to be a CSV or VCF.")
    #executionTime = (time.time() - startTime)
    #print('Execution time in seconds: ' + str(executionTime))


if __name__ == '__main__':
    main()
