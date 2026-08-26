"""Audit that issuer and time identifiers are absent from masked inputs."""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_cninfo_prompt_inputs import mask_identity_and_time, SUBJECT_MASK, TIME_MASK


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--size', type=int, default=50)
    p.add_argument('--seed', type=int, default=20260806); args = p.parse_args()
    rng = random.Random(args.seed); sample=[]; seen=0; failures=[]
    sources = sorted(args.input.glob('part-*.jsonl')) if args.input.is_dir() else [args.input]
    if not sources: raise ValueError(f'no input JSONL files found: {args.input}')
    for source in sources:
        with source.open(encoding='utf-8') as h:
            for line in h:
                if not line.strip(): continue
                r=json.loads(line); seen += 1
                if 'text_input_5_masked_short' in r:
                    short=str(r.get('text_input_5_masked_short','') or ''); long=str(r.get('text_input_6_masked_long','') or '')
                    item={'row_index':r.get('row_index'),'document_id':r.get('document_id'),
                          'masked_short_preview':short[:2500],'masked_long_preview':long[:2500],
                          'subject_mask_count':short.count(SUBJECT_MASK)+long.count(SUBJECT_MASK),
                          'time_mask_count':short.count(TIME_MASK)+long.count(TIME_MASK),
                          'sha256_short':r.get('text_input_5_masked_short_sha256'),'sha256_long':r.get('text_input_6_masked_long_sha256')}
                    if len(sample)<args.size: sample.append(item)
                    elif rng.randrange(seen)<args.size: sample[rng.randrange(args.size)]=item
                    continue
                name=str(r.get('stock_name','') or ''); sid=str(r.get('stock_id','') or '')
                text=str(r.get('text_model',r.get('text','')) or ''); masked=mask_identity_and_time(text,name,sid)
                name_remaining=bool(name and name.replace(' ','') in masked.replace(' ',''))
                code_remaining=bool(sid and re.search(rf'(?<!\d){re.escape(sid)}(?!\d)',masked))
                if name_remaining or code_remaining: failures.append({'stock_id':sid,'name_remaining':name_remaining,'code_remaining':code_remaining})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'input_records':seen,'sample_size':len(sample),'failures':failures,'samples':sample},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'input_records':seen,'sample_size':len(sample),'direct_identifier_failures':len(failures),'output':str(args.output)},ensure_ascii=False))

if __name__=='__main__': main()
