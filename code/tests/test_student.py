"""Contract checks for the new within-window mechanism; official tests unchanged."""
import unittest
import torch
from student import RotaryGPT as rope, build_model

class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config=dict(vocab=2048,width=32,heads=4,depth=2,context=256,copy_alpha=.35,copy_smoothing=2.,copy_order=3)
        self.model=build_model(self.config).eval()

    def test_no_future_or_cross_example_information(self):
        x=torch.randint(0,16,(2,256))
        y=x.clone(); y[:,81:]=(y[:,81:]+7)%16
        with torch.no_grad():
            a=self.model.predict_log_probs(x)
            b=self.model.predict_log_probs(y)
            single=self.model.predict_log_probs(x[:1])
            again=self.model.predict_log_probs(x)
        torch.testing.assert_close(a[:,:81],b[:,:81],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(a[:1],single,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(a,again,atol=0,rtol=0)
        torch.testing.assert_close(a.logsumexp(-1),torch.zeros(2,256),atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.isfinite(a).all())

    def test_copy_only_observed_continuations(self):
        # At final token 2, longest prior suffix [1,2] was followed by 9.
        x=torch.tensor([[1,2,9,7,1,2]])
        p,n=self.model.copy_distribution(x)
        self.assertEqual(n[0,-1,0].item(),1)
        self.assertEqual(p[0,-1,9].item(),1)
        self.assertEqual(n[0,0,0].item(),0)
        # A token after the current position may not become an answer.
        self.assertEqual(n[0,1,0].item(),0)

    def test_disabled_copy_is_exact_neural_ablation(self):
        original=rope(self.config).eval()
        copied=build_model(self.config|{'copy_alpha':0.}).eval()
        copied.load_state_dict(original.state_dict())
        x=torch.randint(0,2048,(2,17))
        with torch.no_grad():
            torch.testing.assert_close(original.predict_log_probs(x),copied.predict_log_probs(x),rtol=0,atol=0)

    def test_single_token_fallback(self):
        x=torch.tensor([[1]])
        with torch.no_grad():
            p=self.model.predict_log_probs(x)
        self.assertEqual(tuple(p.shape),(1,1,2048))
        torch.testing.assert_close(p.logsumexp(-1),torch.zeros(1,1),atol=1e-6,rtol=1e-6)
