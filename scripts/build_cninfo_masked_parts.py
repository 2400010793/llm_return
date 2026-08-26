"""Build the two anonymous prompt inputs as small atomic JSONL parts."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_cninfo_prompt_inputs import (  # noqa: E402
    LONG_TEMPLATE, MASKED_LONG_TEMPLATE, MASKED_SHORT_TEMPLATE, SHORT_TEMPLATE,
    SHORT_PROMPT, LONG_PROMPT,
    mask_identity_and_time, sha,
)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--part-size', type=int, default=5000)
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    args = p.parse_args()
    if args.part_size < 1: raise ValueError('part-size must be positive')
    if not 0 <= args.shard_id < args.num_shards: raise ValueError('invalid shard-id/num-shards')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts=[]; total=0; part=0; in_part=0
    def open_part(index):
        final=args.output_dir/f'part-{index:05d}.jsonl'
        temp=final.with_suffix(final.suffix+f'.tmp.{os.getpid()}')
        return final,temp,temp.open('w',encoding='utf-8')
    final,temp,out=open_part(part)
    try:
        with args.input.open(encoding='utf-8') as source:
            for line in source:
                if not line.strip(): continue
                source_index = total
                total += 1
                if source_index % args.num_shards != args.shard_id: continue
                r=json.loads(line); text=str(r.get('text_model',r.get('text','')) or '')
                title=str(r.get('title_clean_final',r.get('title','')) or '')
                name=str(r.get('stock_name','') or str(r.get('stock_id',''))); sid=str(r.get('stock_id',''))
                short_plain=SHORT_TEMPLATE.format(stock_name=name,text=text)
                long_plain=LONG_TEMPLATE.format(stock_name=name,stock_id=sid,title=title,text=text)
                mt=mask_identity_and_time(text,name,sid); mh=mask_identity_and_time(title,name,sid)
                short=MASKED_SHORT_TEMPLATE.format(text=mt); long=MASKED_LONG_TEMPLATE.format(title=mh,text=mt)
                fixed_long=LONG_PROMPT+title+'\n公告正文:'+text
                fixed_masked_long=LONG_PROMPT+mh+'\n公告正文:'+mt
                row={'row_index':total,'document_id':r.get('document_id'),'stock_id_alignment':sid,
                     'announcement_date_alignment':r.get('announcement_date'),'text_input_5_masked_short':short,
                     'text_input_6_masked_long':long,'text_input_5_masked_short_sha256':sha(short),
                     'text_input_6_masked_long_sha256':sha(long),'text_plain':text,
                     'text_input_2_short':short_plain,'text_input_3_long':long_plain,
                     'text_plain_sha256':sha(text),'text_input_2_short_sha256':sha(short_plain),
                     'text_input_3_long_sha256':sha(long_plain),
                     'short_prompt': SHORT_PROMPT,
                     'short_title': title, 'short_body': text,
                     'masked_short_prompt': SHORT_PROMPT,
                     'masked_short_title': mh, 'masked_short_body': mt,
                     'long_prompt': LONG_PROMPT, 'long_title': title, 'long_body': text,
                     'masked_long_prompt': LONG_PROMPT, 'masked_long_title': mh, 'masked_long_body': mt,
                     'text_input_7_fixed_long': fixed_long, 'text_input_8_fixed_masked_long': fixed_masked_long,
                     'prompt_version':'cninfo_prompt_v2_masked_5_6_all_inputs'}
                out.write(json.dumps(row,ensure_ascii=False)+'\n'); in_part+=1
                if in_part == args.part_size:
                    out.close(); os.replace(temp,final); parts.append({'path':str(final),'rows':in_part})
                    part+=1; in_part=0; final,temp,out=open_part(part)
        out.close()
        if in_part:
            os.replace(temp,final); parts.append({'path':str(final),'rows':in_part})
        else: temp.unlink(missing_ok=True)
    finally:
        if not out.closed: out.close()
    manifest={'input':str(args.input),'source_rows':total,'rows':sum(x['rows'] for x in parts),
              'part_size':args.part_size,'shard_id':args.shard_id,'num_shards':args.num_shards,'parts':parts,
              'prompt_version':'cninfo_prompt_v2_masked_5_6',
              'sha256':hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest()}
    tmp=args.manifest.with_suffix(args.manifest.suffix+f'.tmp.{os.getpid()}'); tmp.parent.mkdir(parents=True,exist_ok=True)
    tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(tmp,args.manifest)
    print(json.dumps({'rows':total,'parts':len(parts),'manifest':str(args.manifest)},ensure_ascii=False))

if __name__ == '__main__': main()