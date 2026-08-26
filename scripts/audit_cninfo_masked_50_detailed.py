"""Detailed random-50 audit of masking and final-text cleanliness."""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.build_cninfo_prompt_inputs import mask_identity_and_time, SUBJECT_MASK, TIME_MASK, DATE_PATTERNS, COMPANY_NAME

BAD_RE=re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ue000-\uf8ff\ufffd]')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--json-output',type=Path,required=True); p.add_argument('--txt-output',type=Path,required=True); p.add_argument('--size',type=int,default=50); p.add_argument('--seed',type=int,default=20260807); a=p.parse_args()
    rng=random.Random(a.seed); sample=[]; n=0
    with a.input.open(encoding='utf-8') as h:
        for line in h:
            if not line.strip(): continue
            r=json.loads(line); n+=1
            if len(sample)<a.size: sample.append(r)
            elif rng.randrange(n)<a.size: sample[rng.randrange(a.size)]=r
    results=[]; failures=[]
    for i,r in enumerate(sample,1):
        text=str(r.get('text_model',r.get('text','')) or ''); title=str(r.get('title_clean_final',r.get('title','')) or '')
        name=str(r.get('stock_name','') or ''); sid=str(r.get('stock_id','') or '')
        masked_text=mask_identity_and_time(text,name,sid); masked_title=mask_identity_and_time(title,name,sid)
        masked=masked_title+'\n'+masked_text
        name_left=bool(name and name.replace(' ','') in masked.replace(' ',''))
        code_left=bool(sid and re.search(rf'(?<!\d){re.escape(sid)}(?!\d)',masked))
        company_left=bool(COMPANY_NAME.search(masked))
        date_left=any(pattern.search(masked) for pattern in DATE_PATTERNS)
        bad=list(BAD_RE.finditer(text)); bad_mask=list(BAD_RE.finditer(masked))
        item={'sample_no':i,'row_index':r.get('row_index'),'document_id':r.get('document_id'),'stock_id':sid,'stock_name':name,'announcement_date':r.get('announcement_date'),'title_original':title,'original_chars':len(text),'masked_chars':len(masked_text),'mask_subject_count':masked.count(SUBJECT_MASK),'mask_time_count':masked.count(TIME_MASK),'name_remaining':name_left,'code_remaining':code_left,'company_legal_name_remaining':company_left,'date_remaining':date_left,'original_dirty_char_count':len(bad),'masked_dirty_char_count':len(bad_mask),'masked_title':masked_title,'masked_text_preview':masked_text[:1800]}
        results.append(item)
        if any((name_left,code_left,company_left,date_left,len(bad),len(bad_mask))): failures.append(item)
    summary={'input_records':n,'sample_size':len(sample),'seed':a.seed,'mask_symbols':{'subject':SUBJECT_MASK,'time':TIME_MASK},'sample_failures':len(failures),'all_sample_identifiers_masked':not any(x['name_remaining'] or x['code_remaining'] or x['company_legal_name_remaining'] or x['date_remaining'] for x in results),'all_sample_text_clean':not any(x['original_dirty_char_count'] or x['masked_dirty_char_count'] for x in results),'records':results}
    a.json_output.parent.mkdir(parents=True,exist_ok=True); a.txt_output.parent.mkdir(parents=True,exist_ok=True)
    a.json_output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with a.txt_output.open('w',encoding='utf-8') as out:
        out.write(f"随机种子: {a.seed}\n输入记录: {n}\n抽样数: {len(sample)}\n失败记录数: {len(failures)}\n")
        out.write(f"身份/代码/公司名/日期全部遮蔽: {summary['all_sample_identifiers_masked']}\n文本清洁检查通过: {summary['all_sample_text_clean']}\n占位符: {SUBJECT_MASK} / {TIME_MASK}\n\n")
        for x in results:
            out.write(f"===== SAMPLE {x['sample_no']} =====\n股票: {x['stock_name']} ({x['stock_id']})  日期: {x['announcement_date']}  row_index: {x['row_index']}\n标题原文: {x['title_original']}\n掩码标题: {x['masked_title']}\n检查: name={x['name_remaining']} code={x['code_remaining']} company={x['company_legal_name_remaining']} date={x['date_remaining']} dirty_original={x['original_dirty_char_count']} dirty_masked={x['masked_dirty_char_count']}\n正文掩码预览:\n{x['masked_text_preview']}\n\n")
    print(json.dumps({'input_records':n,'sample_size':len(sample),'failures':len(failures),'json_output':str(a.json_output),'txt_output':str(a.txt_output)},ensure_ascii=False))
if __name__=='__main__': main()
