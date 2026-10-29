"""Independent bounded arithmetic residual; conceptual sources recorded separately."""
import ast,re
from fractions import Fraction
from pathlib import Path
import numpy as np
from common import parse_answer
from answer_ranker import build_groups

def value(expr):
    if len(expr)>128 or not re.fullmatch(r'[\d.()+*/\-\s]+',expr):raise ValueError('unsupported')
    tree=ast.parse(expr.strip(),mode='eval')
    if sum(1 for _ in ast.walk(tree))>48:raise ValueError('complex')
    def visit(n):
        if isinstance(n,ast.Expression):return visit(n.body)
        if isinstance(n,ast.Constant) and type(n.value) in (float,int):
            v=Fraction(ast.get_source_segment(expr.strip(),n))
        elif isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            v=visit(n.operand)*(-1 if isinstance(n.op,ast.USub) else 1)
        elif isinstance(n,ast.BinOp):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):v=a+b
            elif isinstance(n.op,ast.Sub):v=a-b
            elif isinstance(n.op,ast.Mult):v=a*b
            elif isinstance(n.op,ast.Div):v=a/b
            else:raise ValueError('operator')
        else:raise ValueError('node')
        if abs(v)>10**12 or abs(v.numerator)>10**18 or v.denominator>10**18:raise ValueError('bound')
        return v
    return visit(tree)

def close(a,b):return abs(a-b)<=Fraction(1,200)+Fraction(1,1000000)*max(abs(a),abs(b))

def equations(text):
    s=text[:20000].replace('×','*').replace('÷','/').replace('−','-')
    s=re.sub(r'\*\*([^*\n]+)\*\*',r'\1',s)
    for old,new in [(r'\times','*'),(r'\cdot','*'),(r'\div','/'),(r'\left',''),(r'\right','')]:s=s.replace(old,new)
    for _ in range(3):s=re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',s)
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',s)
    s=s.replace('$',' ').replace(r'\(',' ').replace(r'\)',' ').replace(r'\[',' ').replace(r'\]',' ')
    checks=[]
    for line in s.splitlines():
        line=re.sub(r'^\s*[-*+]\s+','',line)
        parts=line.split('=')
        for left,right in zip(parts,parts[1:]):
            if len(checks)>=48:return checks
            lm=re.search(r'[\d.()+*/\-\s]+$',left);rm=re.match(r'[\d.()+*/\-\s]+',right)
            if lm is None or rm is None:continue
            before=left[:lm.start()].rstrip();after=right[rm.end():]
            if before and (before[-1].isalnum() or before[-1] in '_%}'):continue
            if after and (after[0] in '%_\\{' or (after[0].isalpha() and not rm.group()[-1].isspace())):continue
            lhs=lm.group().strip();rhs=rm.group().strip().rstrip('.')
            try:checks.append((value(lhs),value(rhs),int(any(c in lhs for c in '+*/-'))))
            except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):continue
    return checks

def evidence(bank,groups):
    output=np.zeros((len(bank),4,48))
    for i,row in enumerate(bank):
        answers=[Fraction(a) for a,_ in groups[i]];cand=[]
        for c in row['candidates']:
            text=c['text'];checks=equations(text);ans=parse_answer(text);a=Fraction(ans) if ans is not None else None
            mismatch=[not close(l,r) for l,r,_ in checks]
            bad=sum(mismatch);count=len(checks)
            last=checks[-1] if checks else None
            own=sum(a is not None and close(l,a) for l,r,_ in checks)
            other=sum(any(close(l,b) for b in answers if b!=a) for l,r,_ in checks)
            cand.append([float('=' in text),np.log1p(text.count('=')),np.log1p(sum('=' in ln for ln in text.splitlines())),
                float(count>0),np.log1p(count),np.log1p(bad),bad/max(count,1),
                max([float(abs(l-r)/max(abs(l),abs(r),1)) for l,r,_ in checks],default=0),
                float(last is not None and a is not None),float(last is not None and a is not None and close(last[0],a)),
                float(last is not None and a is not None and close(last[1],a)),
                float(last is not None and not close(last[0],last[1])),np.log1p(own),np.log1p(other),
                sum(flag for l,r,flag in checks)/max(count,1),float(a is not None and a.denominator==1)])
        cand=np.asarray(cand)
        for j,(_,members) in enumerate(groups[i]):
            f=cand[members];output[i,j]=np.r_[f.mean(0),f.min(0),f.max(0)]
    return output

def predict(bank,arrays,checkpoint):
    a=np.load(Path(checkpoint)/'model.npz',allow_pickle=False)
    x,m,g=build_groups(bank,arrays,'numeric');f=evidence(bank,g)
    s=x@a['linear']+f@a['correction'];s[~m]=-np.inf
    return [int(min(row[int(s[i].argmax())][1],key=lambda j:bank[i]['candidates'][j]['text'])) if row else 0 for i,row in enumerate(g)]
