"""Verify native observations, commands, labels and shard completeness."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from reacher_task import DATA, EPISODES, TRAIN_STEPS, Reacher, applied, score, success


def verify(collection):
    env=Reacher()
    rows=[]
    assert float(score([0.,0.])) == 1 and success([2*np.pi, 0.])
    assert not success([.051, 0.])
    for shift in ("original","joint-phase","harmonic"):
        np.testing.assert_allclose(applied([0.2,0.3],[0.,0.],shift),[0.2,0.3],atol=1e-7)
        env.reset(100000)
        p0=env.position;img=env.render()
        assert img.dtype==np.uint8 and img.shape==(224,224,3) and img.std()>0
        command=np.array([0.4,-0.2],np.float32)
        real_command=applied(command,p0,shift)
        q=env.step(command,shift)
        env.reset(100000)
        np.testing.assert_array_equal(env.position,p0)
        q_native=env.step(real_command,"original")
        np.testing.assert_array_equal(q_native,q)
        rows.append({'shift':shift,'action_sent':command.tolist(),
                     'applied_at_initial_state':real_command.tolist(),
                     'initial_q':p0.tolist(),'after_q':q.tolist(),
                     'after_score':float(score(q))})
    report={'native_rgb_shape':list(img.shape),'native_rgb_dtype':str(img.dtype),
            'native_render_std':float(img.std()),'field_checks':rows}
    if collection:
        info=json.loads((DATA/"complete.json").read_text())
        assert info["rows"] == EPISODES*TRAIN_STEPS
        a=np.load(DATA/"actions.npy",mmap_mode="r")
        p=np.load(DATA/"qpos.npy",mmap_mode="r")
        pixels=np.load(DATA/"pixels.npy",mmap_mode="r")
        checked=[]
        for episode in (0,1001,2999):
            env.reset(100000+episode)
            policy=np.random.default_rng(130000+episode)
            for t in range(TRAIN_STEPS):
                cmd=policy.uniform(-1,1,2).astype(np.float32)
                np.testing.assert_array_equal(a[episode,t],cmd)
                np.testing.assert_array_equal(p[episode,t],env.position.astype(np.float32))
                if t in (0,5,99,199):
                    np.testing.assert_array_equal(env.render(),pixels[episode,t])
                    checked.append([episode,t])
                env.step(cmd,"original")
        assert np.isfinite(p).all() and (np.abs(a)<=1).all()
        report['collection']=info
        report['replayed_episodes']=[0,1001,2999]
        report['bit_exact_rgb_sample_indices']=checked
        report['checksums']={}
        for name in ('actions.npy','qpos.npy','pixels.npy'):
            with (DATA/name).open('rb') as source:
                report['checksums'][name]=hashlib.file_digest(source,'sha256').hexdigest()
    env.close()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(verify(args.collection),indent=2,allow_nan=False)+"\n")
