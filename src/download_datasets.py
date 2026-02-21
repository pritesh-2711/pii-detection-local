"""
PII Dataset Collection - Banking Domain

Collects and inspects all PII/NER datasets relevant to banking.

Datasets covered:
1. ai4privacy/pii-masking-400k
2. ai4privacy/pii-masking-300k (FinPII-80k split)
3. gretelai/synthetic_pii_finance_multilingual
4. nvidia/Nemotron-PII
5. wikiann (en)
6. Babelscape/multinerd (en)
7. DFKI-SLT/few-nerd
8. conll2003
9. nlpaueb/finer-139
10. Isotonic/pii-masking-200k

Usage:
    python src/download_datasets.py
    python src/download_datasets.py --output-dir ./pii_datasets
"""

import os
import json
import warnings
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset, get_dataset_config_names
from tabulate import tabulate

warnings.filterwarnings('ignore')


def count_labels_from_bio(dataset, label_field='ner_tags', label_names=None):
    types = set()
    sample = dataset.select(range(min(500, len(dataset))))
    for row in sample:
        tags = row[label_field]
        for tag in tags:
            if isinstance(tag, int):
                if label_names:
                    tag = label_names[tag]
                else:
                    continue
            if tag != 'O' and tag != '':
                entity = tag.replace('B-', '').replace('I-', '').strip()
                if entity:
                    types.add(entity)
    return types


def save_split(dataset, name, split_name, output_dir: Path):
    out_path = output_dir / name
    out_path.mkdir(exist_ok=True)
    filepath = out_path / f'{split_name}.jsonl'
    dataset.to_json(str(filepath))
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f'  Saved {split_name}: {len(dataset):,} rows -> {filepath} ({size_mb:.1f} MB)')
    return filepath


def build_summary_row(name, url, license_, lang, num_rows, num_entity_types,
                      entity_types, annotation_source, domain, banking_relevance, notes):
    return {
        'Dataset': name,
        'URL': url,
        'License': license_,
        'Language(s)': lang,
        'Total Rows': num_rows,
        'PII Entity Types (#)': num_entity_types,
        'Entity Types (sample)': entity_types,
        'Annotation Source': annotation_source,
        'Domain': domain,
        'Banking Relevance': banking_relevance,
        'Notes': notes,
    }


def main(output_dir: str = './pii_datasets'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f'Output directory: {output_dir.resolve()}')

    summary_rows = []

    # ------------------------------------------------------------------
    # Dataset 1 — ai4privacy/pii-masking-400k
    # ------------------------------------------------------------------
    print('Loading ai4privacy/pii-masking-400k ...')
    ds_400k = load_dataset('ai4privacy/pii-masking-400k')
    print(ds_400k)

    train_split = ds_400k['train']
    bio_types = set()
    for row in train_split.select(range(min(1000, len(train_split)))):
        for label in row.get('bio_labels', []):
            if label != 'O':
                bio_types.add(label.replace('B-', '').replace('I-', ''))

    print(f'  Rows: {len(train_split):,}')
    print(f'  Detected entity types ({len(bio_types)}): {sorted(bio_types)}')
    save_split(train_split, 'ai4privacy_400k', 'train', output_dir)

    summary_rows.append(build_summary_row(
        name='ai4privacy/pii-masking-400k',
        url='https://huggingface.co/datasets/ai4privacy/pii-masking-400k',
        license_='Custom (academic free, commercial needs license)',
        lang='en, fr, de, it',
        num_rows=len(train_split),
        num_entity_types=63,
        entity_types='FIRSTNAME, LASTNAME, EMAIL, PHONE, CREDITCARDNUMBER, SSN, IBAN, BITCOINADDRESS, ...',
        annotation_source='Synthetic (proprietary algorithm)',
        domain='General (business, education, psychology, legal)',
        banking_relevance='High — covers IBAN, CREDITCARD, ACCOUNTNUMBER, BITCOINADDRESS',
        notes='Latest version. 63 PII classes. Use FinPII split in 300k for finance-specific.',
    ))

    # ------------------------------------------------------------------
    # Dataset 2 — ai4privacy/pii-masking-300k
    # ------------------------------------------------------------------
    print('Loading ai4privacy/pii-masking-300k ...')
    try:
        configs = get_dataset_config_names('ai4privacy/pii-masking-300k')
        print(f'  Available configs: {configs}')
    except Exception:
        configs = ['default']

    ds_300k = load_dataset('ai4privacy/pii-masking-300k')
    print(ds_300k)

    train_300k = ds_300k['train']
    bio_types_300k = set()
    for row in train_300k.select(range(min(1000, len(train_300k)))):
        for label in row.get('bio_labels', []):
            if label != 'O':
                bio_types_300k.add(label.replace('B-', '').replace('I-', ''))

    print(f'  Rows: {len(train_300k):,}')
    print(f'  Detected entity types ({len(bio_types_300k)}): {sorted(bio_types_300k)}')
    save_split(train_300k, 'ai4privacy_300k', 'train', output_dir)

    summary_rows.append(build_summary_row(
        name='ai4privacy/pii-masking-300k',
        url='https://huggingface.co/datasets/ai4privacy/pii-masking-300k',
        license_='Custom (academic free, commercial needs license)',
        lang='en, fr, de, it, es, pt',
        num_rows=len(train_300k),
        num_entity_types='27 (OpenPII) + ~20 (FinPII)',
        entity_types='OpenPII-220k + FinPII-80k (finance/insurance-specific types)',
        annotation_source='Synthetic + human-in-loop (~98.3% token accuracy)',
        domain='General + Finance/Insurance (FinPII subset)',
        banking_relevance='Very High — FinPII-80k explicitly targets finance/insurance',
        notes='Best option for banking. FinPII contains ~20 finance-specific entity types.',
    ))

    # ------------------------------------------------------------------
    # Dataset 3 — gretelai/synthetic_pii_finance_multilingual
    # ------------------------------------------------------------------
    print('Loading gretelai/synthetic_pii_finance_multilingual ...')
    ds_gretel = load_dataset('gretelai/synthetic_pii_finance_multilingual')
    print(ds_gretel)

    train_gretel = ds_gretel['train']
    print(f'  Rows: {len(train_gretel):,}')
    print(f'  Columns: {train_gretel.column_names}')

    pii_types_gretel = set()
    label_col = None
    for col in ['pii_class', 'entity_type', 'label', 'ner_tags', 'labels']:
        if col in train_gretel.column_names:
            label_col = col
            break

    if label_col:
        for row in train_gretel.select(range(min(200, len(train_gretel)))):
            val = row[label_col]
            if isinstance(val, list):
                for v in val:
                    pii_types_gretel.add(str(v))
            else:
                pii_types_gretel.add(str(val))

    print(f'  PII types found: {sorted(pii_types_gretel)}')
    save_split(train_gretel, 'gretel_finance', 'train', output_dir)
    if 'test' in ds_gretel:
        save_split(ds_gretel['test'], 'gretel_finance', 'test', output_dir)

    summary_rows.append(build_summary_row(
        name='gretelai/synthetic_pii_finance_multilingual',
        url='https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual',
        license_='Apache 2.0',
        lang='en, es, sv, de, it, nl, fr',
        num_rows=len(train_gretel),
        num_entity_types=29,
        entity_types='ACCOUNT_NUMBER, ROUTING_NUMBER, IBAN, CREDIT_CARD, SSN, TAX_ID, ...',
        annotation_source='Synthetic (Gretel LLM + GLiNER validation + LLM-as-judge)',
        domain='Finance (100 financial document types: bank statements, loan docs, wire transfers)',
        banking_relevance='Very High — explicitly covers banking document formats',
        notes='55,940 records. Avg doc length 1,357 chars. Best domain match for banking.',
    ))

    # ------------------------------------------------------------------
    # Dataset 4 — nvidia/Nemotron-PII
    # ------------------------------------------------------------------
    print('Loading nvidia/Nemotron-PII ...')
    ds_nvidia = load_dataset('nvidia/Nemotron-PII')
    print(ds_nvidia)

    train_nvidia = ds_nvidia['train']
    print(f'  Rows: {len(train_nvidia):,}')
    print(f'  Columns: {train_nvidia.column_names}')

    nvidia_types = set()
    for row in train_nvidia.select(range(min(500, len(train_nvidia)))):
        for col in ['ner_tags', 'labels', 'bio_labels', 'label']:
            if col in row:
                val = row[col]
                if isinstance(val, list):
                    for v in val:
                        tag = str(v)
                        if tag != 'O':
                            nvidia_types.add(tag.replace('B-', '').replace('I-', ''))
                break

    print(f'  Entity types: {sorted(nvidia_types)}')
    save_split(train_nvidia, 'nvidia_nemotron', 'train', output_dir)

    summary_rows.append(build_summary_row(
        name='nvidia/Nemotron-PII',
        url='https://huggingface.co/datasets/nvidia/Nemotron-PII',
        license_='CC-BY 4.0',
        lang='en',
        num_rows=len(train_nvidia),
        num_entity_types='55+',
        entity_types='PII + PHI: names, SSN, DOB, ACCOUNT, DEVICE_ID, IP, BIOMETRIC, ...',
        annotation_source='Synthetic (NVIDIA NeMo Data Designer, Census-grounded personas)',
        domain='General across 50+ industries including finance',
        banking_relevance='High — 50+ industries includes finance; covers PHI useful for KYC',
        notes='100k records. Structured + unstructured docs. CC-BY 4.0 = commercial-friendly.',
    ))

    # ------------------------------------------------------------------
    # Dataset 5 — wikiann (en)
    # ------------------------------------------------------------------
    print('Loading wikiann (en) ...')
    ds_wikiann = load_dataset('wikiann', 'en')
    print(ds_wikiann)

    train_wiki = ds_wikiann['train']
    label_names = train_wiki.features['ner_tags'].feature.names
    print(f'  Labels: {label_names}')
    print(f'  Rows: {len(train_wiki):,}')

    save_split(train_wiki, 'wikiann', 'train', output_dir)
    save_split(ds_wikiann['validation'], 'wikiann', 'validation', output_dir)
    save_split(ds_wikiann['test'], 'wikiann', 'test', output_dir)

    entity_types_wiki = set(l.replace('B-', '').replace('I-', '') for l in label_names if l != 'O')

    summary_rows.append(build_summary_row(
        name='wikiann (en)',
        url='https://huggingface.co/datasets/wikiann',
        license_='CC-BY-SA 3.0',
        lang='en (282 langs available)',
        num_rows=len(train_wiki) + len(ds_wikiann['validation']) + len(ds_wikiann['test']),
        num_entity_types=len(entity_types_wiki),
        entity_types=', '.join(sorted(entity_types_wiki)),
        annotation_source='Auto-annotated from Wikipedia using cross-lingual projection',
        domain='General (Wikipedia articles)',
        banking_relevance='Medium — PER/ORG/LOC only; good for entity grounding in text',
        notes='Only 3 entity types. Use as supplementary data for PER/ORG/LOC coverage.',
    ))

    # ------------------------------------------------------------------
    # Dataset 6 — Babelscape/multinerd (en)
    # ------------------------------------------------------------------
    print('Loading Babelscape/multinerd ...')
    ds_multinerd = load_dataset('Babelscape/multinerd', verification_mode='no_checks')
    print(ds_multinerd)

    train_mn = ds_multinerd['train']
    if 'lang' in train_mn.column_names:
        train_mn_en = train_mn.filter(lambda x: x['lang'] == 'en')
    else:
        train_mn_en = train_mn

    multinerd_id2label = {
        0: 'O',
        1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
        5: 'B-LOC', 6: 'I-LOC', 7: 'B-ANIM', 8: 'I-ANIM',
        9: 'B-BIO', 10: 'I-BIO', 11: 'B-CEL', 12: 'I-CEL',
        13: 'B-DIS', 14: 'I-DIS', 15: 'B-EVE', 16: 'I-EVE',
        17: 'B-FOOD', 18: 'I-FOOD', 19: 'B-INST', 20: 'I-INST',
        21: 'B-MEDIA', 22: 'I-MEDIA', 23: 'B-MYTH', 24: 'I-MYTH',
        25: 'B-PLANT', 26: 'I-PLANT', 27: 'B-TIME', 28: 'I-TIME',
        29: 'B-VEHI', 30: 'I-VEHI',
    }
    entity_types_mn = set(
        v.replace('B-', '').replace('I-', '')
        for v in multinerd_id2label.values() if v != 'O'
    )
    print(f'  Labels: {sorted(entity_types_mn)}')
    print(f'  EN rows: {len(train_mn_en):,}')

    save_split(train_mn_en, 'multinerd', 'train_en', output_dir)

    summary_rows.append(build_summary_row(
        name='Babelscape/multinerd',
        url='https://huggingface.co/datasets/Babelscape/multinerd',
        license_='CC-BY-NC-SA 4.0',
        lang='en, de, es, fr, it, nl, pl, pt, ru, zh',
        num_rows=len(train_mn_en),
        num_entity_types=len(entity_types_mn),
        entity_types=', '.join(sorted(entity_types_mn)),
        annotation_source='Expert-annotated',
        domain='General (Wikipedia + news)',
        banking_relevance='Medium — PER/ORG/LOC plus TIME/EVE useful for transaction context',
        notes='15 types. NC license — not for commercial use as-is.',
    ))

    # ------------------------------------------------------------------
    # Dataset 7 — DFKI-SLT/few-nerd
    # ------------------------------------------------------------------
    print('Loading DFKI-SLT/few-nerd (supervised split) ...')
    ds_fewnerd = load_dataset('DFKI-SLT/few-nerd', 'supervised')
    print(ds_fewnerd)

    train_fn = ds_fewnerd['train']
    label_names_fn = train_fn.features['ner_tags'].feature.names

    fn_types = set()
    for row in train_fn.select(range(min(500, len(train_fn)))):
        for tag_id in row['ner_tags']:
            label = label_names_fn[tag_id]
            if label != 'O':
                fn_types.add(label.replace('B-', '').replace('I-', ''))

    print(f'  Rows: {len(train_fn):,}')
    print(f'  Entity types ({len(fn_types)}): {sorted(fn_types)[:20]} ...')

    save_split(train_fn, 'few_nerd', 'train', output_dir)
    save_split(ds_fewnerd['validation'], 'few_nerd', 'validation', output_dir)
    save_split(ds_fewnerd['test'], 'few_nerd', 'test', output_dir)

    summary_rows.append(build_summary_row(
        name='DFKI-SLT/few-nerd',
        url='https://huggingface.co/datasets/DFKI-SLT/few-nerd',
        license_='CC-BY-SA 4.0',
        lang='en',
        num_rows=len(train_fn) + len(ds_fewnerd['validation']) + len(ds_fewnerd['test']),
        num_entity_types=66,
        entity_types='person-politician, org-company, location-city, product-software, ... (66 fine-grained)',
        annotation_source='Crowdsourced (188k sentences)',
        domain='General (Wikipedia)',
        banking_relevance='Low-Medium — org-company, location useful; no direct PII types',
        notes='Fine-grained NER. Useful for entity disambiguation, not direct PII tagging.',
    ))

    # ------------------------------------------------------------------
    # Dataset 8 — CoNLL-2003
    # ------------------------------------------------------------------
    print('Loading conll2003 ...')
    ds_conll = load_dataset('conll2003', revision='refs/convert/parquet')
    print(ds_conll)

    train_conll = ds_conll['train']
    label_names_conll = train_conll.features['ner_tags'].feature.names
    entity_types_conll = set(l.replace('B-', '').replace('I-', '') for l in label_names_conll if l != 'O')

    print(f'  Rows: {len(train_conll):,}')
    print(f'  Labels: {label_names_conll}')

    save_split(train_conll, 'conll2003', 'train', output_dir)
    save_split(ds_conll['validation'], 'conll2003', 'validation', output_dir)
    save_split(ds_conll['test'], 'conll2003', 'test', output_dir)

    summary_rows.append(build_summary_row(
        name='conll2003',
        url='https://huggingface.co/datasets/conll2003',
        license_='Custom (non-commercial research)',
        lang='en',
        num_rows=len(train_conll) + len(ds_conll['validation']) + len(ds_conll['test']),
        num_entity_types=len(entity_types_conll),
        entity_types=', '.join(sorted(entity_types_conll)),
        annotation_source='Expert-annotated (newswire)',
        domain='News (Reuters 1996)',
        banking_relevance='Low — only PER/ORG/LOC/MISC; no financial PII',
        notes='Industry baseline. Non-commercial license.',
    ))

    # ------------------------------------------------------------------
    # Dataset 9 — nlpaueb/finer-139
    # ------------------------------------------------------------------
    print('Loading nlpaueb/finer-139 ...')
    ds_finer = load_dataset('nlpaueb/finer-139', revision='refs/convert/parquet')
    print(ds_finer)

    train_finer = ds_finer['train']
    label_names_finer = train_finer.features['ner_tags'].feature.names

    finer_types = set()
    for row in train_finer.select(range(min(1000, len(train_finer)))):
        for tag_id in row['ner_tags']:
            label = label_names_finer[tag_id]
            if label != 'O':
                finer_types.add(label.replace('B-', '').replace('I-', ''))

    print(f'  Rows: {len(train_finer):,}')
    print(f'  Entity type count: {len(label_names_finer)} labels ({len(finer_types)} found in sample)')
    print(f'  Sample types: {sorted(list(finer_types))[:10]}')

    save_split(train_finer, 'finer_139', 'train', output_dir)
    save_split(ds_finer['validation'], 'finer_139', 'validation', output_dir)
    save_split(ds_finer['test'], 'finer_139', 'test', output_dir)

    summary_rows.append(build_summary_row(
        name='nlpaueb/finer-139',
        url='https://huggingface.co/datasets/nlpaueb/finer-139',
        license_='CC-BY-SA 4.0',
        lang='en',
        num_rows=len(train_finer) + len(ds_finer['validation']) + len(ds_finer['test']),
        num_entity_types=139,
        entity_types='XBRL financial tags: Revenue, Assets, LiabilitiesTotal, DebtCurrent, EPS, ...',
        annotation_source='Expert-annotated (SEC professional auditors via EDGAR filings)',
        domain='Finance (SEC 10-K/10-Q annual/quarterly reports)',
        banking_relevance='High for financial domain language; not classical PII but financial entities',
        notes='1.1M sentences. Use for continued pre-training on financial text.',
    ))

    # ------------------------------------------------------------------
    # Dataset 10 — Isotonic/pii-masking-200k
    # ------------------------------------------------------------------
    print('Loading Isotonic/pii-masking-200k ...')
    ds_iso = load_dataset('Isotonic/pii-masking-200k')
    print(ds_iso)

    split_key = list(ds_iso.keys())[0]
    split_iso = ds_iso[split_key]
    print(f'  Rows: {len(split_iso):,}')
    print(f'  Columns: {split_iso.column_names}')

    bio_types_iso = set()
    for row in split_iso.select(range(min(500, len(split_iso)))):
        for label in row.get('bio_labels', []):
            if label != 'O':
                bio_types_iso.add(label.replace('B-', '').replace('I-', ''))

    print(f'  Entity types ({len(bio_types_iso)}): {sorted(bio_types_iso)}')
    save_split(split_iso, 'isotonic_pii_200k', split_key, output_dir)

    summary_rows.append(build_summary_row(
        name='Isotonic/pii-masking-200k',
        url='https://huggingface.co/datasets/Isotonic/pii-masking-200k',
        license_='Apache 2.0',
        lang='en, fr, de, it',
        num_rows=len(split_iso),
        num_entity_types=len(bio_types_iso) if bio_types_iso else 54,
        entity_types=', '.join(sorted(bio_types_iso)) if bio_types_iso else 'Same 54 classes as ai4privacy series',
        annotation_source='Synthetic (ai4privacy pipeline)',
        domain='General',
        banking_relevance='Medium — same PII classes as ai4privacy series; clean Parquet format',
        notes='Clean mirror of ai4privacy/pii-masking-200k. Apache 2.0 license.',
    ))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    df_summary = pd.DataFrame(summary_rows)
    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    print('\n===== DATASET SUMMARY =====')
    print(tabulate(df_summary, headers='keys', tablefmt='grid', showindex=False))

    csv_path = output_dir / 'dataset_summary.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f'\nSummary saved to: {csv_path}')

    # ------------------------------------------------------------------
    # Verify saved files
    # ------------------------------------------------------------------
    print('\n===== FILES ON DISK =====')
    total_size = 0
    file_rows = []
    for jsonl_file in sorted(output_dir.rglob('*.jsonl')):
        size_mb = jsonl_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        file_rows.append({
            'File': str(jsonl_file.relative_to(output_dir)),
            'Size (MB)': f'{size_mb:.1f}',
        })
    print(tabulate(file_rows, headers='keys', tablefmt='simple'))
    print(f'\nTotal disk usage: {total_size:.1f} MB')

    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download PII datasets from HuggingFace')
    parser.add_argument('--output-dir', type=str, default='./pii_datasets',
                        help='Directory to save downloaded datasets')
    args = parser.parse_args()
    main(output_dir=args.output_dir)