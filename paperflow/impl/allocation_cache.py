"""Validate and persist COMPOT spectral allocation results as JSON."""
import json
import math
from pathlib import Path
from .allocation import get_dynamic_cr_svd_all_enhanced_mincr

def load_or_allocate(model, module_names, args):
    keys={f'model.layers.{i}.{name}' for name in module_names for i in range(len(model.model.layers))}
    signature=dict(model=args.model_name, revision=getattr(model.config,'_commit_hash',None),
                   cr=args.compression_ratio, ks_ratio=args.ks_ratio,
                   min_cr=args.allocation_min_cr, max_cr=args.allocation_max_cr,
                   norm=args.allocation_norm, whitening=False)
    path=Path(args.save_path)/'allocation.json'
    if path.exists():
        payload=json.loads(path.read_text())
        if payload['signature'] != signature:
            raise ValueError('Allocation configuration changed; choose a fresh checkpoint root')
        result=payload['allocation']
    else:
        if any(Path(args.save_path).glob('*_u.pth')):
            raise ValueError('Factors without allocation provenance; choose a fresh checkpoint root')
        result=get_dynamic_cr_svd_all_enhanced_mincr(model,module_names,args.compression_ratio,
            normalization=bool(args.allocation_norm), norm=args.allocation_norm,
            min_cr=args.allocation_min_cr,max_cr=args.allocation_max_cr,ks_ratio=args.ks_ratio)
        payload={'signature':signature,'allocation':result}
    if set(result)!=keys or any(not math.isfinite(v) or not 0<=v<1 for v in result.values()):
        raise ValueError('Allocation must contain exactly all model projections with CR in [0,1)')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2));temporary.replace(path)
    return result
