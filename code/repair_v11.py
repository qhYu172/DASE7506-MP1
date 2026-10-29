"""Bounded propagation of explicit arithmetic slips; references never used."""
import re
from fractions import Fraction
from pathlib import Path
import numpy as np
from arithmetic_v10 import value,close,evidence
from common import parse_answer
from answer_ranker import build_groups

TOKEN=re.compile(r'(?<![\w.])\d+(?:\.\d+)?')

def trace(text):
    s=text[:20000].replace('×','*').replace('÷','/').replace('−','-')
    s=re.sub(r'\*\*([^*\n]+)\*\*',r'\1',s)
    for old,new in [(r'\times','*'),(r'\cdot','*'),(r'\div','/'),(r'\left',''),(r'\right','')]:s=s.replace(old,new)
    for _ in range(3):s=re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',s)
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',s)
    for sep in ('$',r'\(',r'\)',r'\[',r'\]'):s=s.replace(sep,' ')
    mapping={};ambiguous=set();rows=[]
    for line in s.splitlines():
        line=re.sub(r'^\s*[-*+]\s+','',line);parts=line.split('=')
        for left,right in zip(parts,parts[1:]):
            if len(rows)>=48:return rows
            lm=re.search(r'[\d.()+*/\-\s]+$',left);rm=re.match(r'[\d.()+*/\-\s]+',right)
            if lm is None or rm is None:continue
            before=left[:lm.start()].rstrip();after=right[rm.end():]
            if before and (before[-1].isalnum() or before[-1] in '_%}'):continue
            if after and (after[0] in '%_\\{' or (after[0].isalpha() and not rm.group()[-1].isspace())):continue
            lhs=lm.group().strip();rhs=rm.group().strip().rstrip('.')
            try:
                a,b=value(lhs),value(rhs);replacements=[]
                def replace(match):
                    old=Fraction(match.group())
                    if old in mapping and old not in ambiguous:
                        replacements.append(old);new=mapping[old]
                        return f'({new.numerator}/{new.denominator})'
                    return match.group()
                fixed=value(TOKEN.sub(replace,lhs))
                rows.append((a,b,fixed,len(replacements)))
                if not close(fixed,b) and re.fullmatch(r'\d+(?:\.\d+)?',rhs):
                    if b in mapping and not close(mapping[b],fixed):ambiguous.add(b)
                    elif b!=0:mapping[b]=fixed
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):continue
    return rows

def extra(bank,groups):
    output=np.zeros((len(bank),4,46))
    for i,row in enumerate(bank):
        aa=[Fraction(a) for a,_ in groups[i]];vectors=[];ends=[]
        for c in row['candidates']:
            ts=trace(c['text']);answer=parse_answer(c['text']);a=Fraction(answer) if answer is not None else None
            n=len(ts);bad=sum(not close(l,r) for l,r,f,k in ts);fixedbad=sum(not close(f,r) for l,r,f,k in ts)
            last=ts[-1] if ts else None;end=last[2] if last else None;ends.append(end)
            changed=sum(not close(l,f) for l,r,f,k in ts)
            vectors.append([np.log1p(n),np.log1p(bad),np.log1p(fixedbad),np.log1p(sum(k for l,r,f,k in ts)),
                np.log1p(changed),bad/max(n,1),fixedbad/max(n,1),float(end is not None),
                float(end is not None and a is not None and close(end,a)),float(end is not None and any(close(end,b) for b in aa if b!=a)),
                float(last is not None and not close(last[0],last[2])),
                float(last is not None and a is not None and close(last[1],a)),
                min(float(abs(end-a)/max(abs(end),abs(a),1)),1) if end is not None and a is not None else 0.,
                float(changed>0)])
        vectors=np.asarray(vectors)
        for j,(ans,members) in enumerate(groups[i]):
            v=vectors[members];a=Fraction(ans)
            own=[ends[k] for k in members if ends[k] is not None];other=[e for k,e in enumerate(ends) if k not in members and e is not None]
            cross=[sum(close(e,a) for e in own)/max(len(own),1),sum(close(e,a) for e in other)/max(len(other),1),float(bool(own)),float(bool(other))]
            output[i,j]=np.r_[v.mean(0),v.min(0),v.max(0),cross]
    return output

def predict(bank,arrays,checkpoint):
    a=np.load(Path(checkpoint)/'model.npz',allow_pickle=False)
    x,m,g=build_groups(bank,arrays,'numeric');f=np.concatenate([evidence(bank,g),extra(bank,g)],axis=2)
    s=x@a['linear']+f@a['correction'];s[~m]=-np.inf
    return [int(min(row[int(s[i].argmax())][1],key=lambda j:bank[i]['candidates'][j]['text'])) if row else 0 for i,row in enumerate(g)]
