import unittest
import numpy as np
import torch
from experiments.relation.tcn_baseline import Segment
from experiments.paper_validation.perturb import plans,view,singleton_fallback,infer,intervene


class Interventions(unittest.TestCase):
    def setUp(self):
        self.segments=[];self.metadata=[]
        for si,n in enumerate((1,2,4,3)):
            ids=[f"{si}_{j}" for j in range(n)]
            a=np.arange(n*7,dtype=np.float32).reshape(n,7)+si*100
            f=np.concatenate([a,a],axis=1)
            self.segments.append(Segment(ids,f,a,np.arange(n)%7,["clinical"]*n))
            for c in ids:self.metadata.append({"clip_id":c,"split":"val","source_kind":"clinical",
                "recording_key":str(si),"source_collection":"same","label_id":0})

    def test_fixed_label_free_sampling(self):
        p=plans(self.segments,self.metadata)
        altered=[{**r,"label_id":6} for r in self.metadata]
        self.assertEqual(p,plans(self.segments,altered))
        self.assertEqual(p,plans(self.segments,self.metadata))

    def test_shuffle_real_derangement(self):
        p=plans(self.segments,self.metadata)
        self.assertEqual(len(p["shuffle"]),7)
        for r in p["shuffle"]:
            for pos,ds,dj in r["swaps"]:
                self.assertEqual(ds,r["segment"]);self.assertNotEqual(pos,dj)
                self.assertNotEqual(pos,r["anchor"]);self.assertNotEqual(dj,r["anchor"])

    def test_anchor_and_donor_pairs(self):
        for r in plans(self.segments,self.metadata)["cross_clinical"]:
            s=view(self.segments,r);orig=self.segments[r["segment"]]
            np.testing.assert_array_equal(s.features[r["anchor"]],orig.features[r["anchor"]])
            np.testing.assert_array_equal(s.a_logits[r["anchor"]],orig.a_logits[r["anchor"]])
            np.testing.assert_array_equal(s.labels,orig.labels)
            for pos,ds,dj in r["swaps"]:
                self.assertNotEqual(ds,r["segment"])
                np.testing.assert_array_equal(s.features[pos],self.segments[ds].features[dj])
                np.testing.assert_array_equal(s.a_logits[pos],self.segments[ds].a_logits[dj])

    def test_refuse_train_or_anchor_change(self):
        with self.assertRaises(ValueError):plans(self.segments,[{**r,"split":"train"} for r in self.metadata])
        with self.assertRaises(ValueError):view(self.segments,{"segment":0,"anchor":0,"swaps":[[0,1,0]]})

    def test_fallback_exact_only_singletons(self):
        z=torch.randn(10,7);a=torch.randn(10,7)
        protected=singleton_fallback(z,a,self.segments)
        self.assertTrue(torch.equal(protected[0],a[0]))
        self.assertTrue(torch.equal(protected[1:],z[1:]))
        self.assertTrue(torch.equal(z,singleton_fallback(z,z,self.segments)))

    def test_identity_forward_interventions(self):
        class Identity(torch.nn.Module):
            def forward(self,f,a,valid):return a
        m=Identity();device=torch.device("cpu")
        z=infer(m,self.segments,device)
        for name in ("shuffle","cross_clinical"):
            changed,_=intervene(m,self.segments,z,plans(self.segments,self.metadata)[name],device)
            self.assertTrue(torch.equal(z,changed))


if __name__=="__main__":unittest.main()
