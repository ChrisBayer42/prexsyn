import torch

from prexsyn.data.struct import PropertyRepr, SynthesisRepr, concat_synthesis_reprs
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.queries import Query, QueryPlanner
from prexsyn_engine.featurizer.synthesis import PostfixNotationTokenDef

from .builder import SynthesisReprBuilder


class QuerySampler:
    def __init__(
        self,
        model: PrexSyn,
        token_def: PostfixNotationTokenDef,
        num_samples: int,
        max_length: int = 16,
    ) -> None:
        super().__init__()
        self.model = model
        self.token_def = token_def
        self.num_samples = num_samples
        self.max_length = max_length

    def _create_builder(self, batch_size: int) -> SynthesisReprBuilder:
        return SynthesisReprBuilder(
            batch_size=batch_size,
            device=self.model.device,
            bb_token=self.token_def.BB,
            rxn_token=self.token_def.RXN,
            pad_token=self.token_def.PAD,
            start_token=self.token_def.START,
            end_token=self.token_def.END,
        )

    def _sample_conjunctive(self, property_repr: PropertyRepr, weight: torch.Tensor) -> SynthesisRepr:
        e_property = self.model.embed_properties(property_repr)
        if self.num_samples > 1:
            e_property = e_property.repeat(self.num_samples)
        batch_size = e_property.batch_size
        builder = self._create_builder(batch_size)
        for _ in range(self.max_length):
            e_synthesis = self.model.embed_synthesis(builder.get())
            h_syn = self.model.encode(e_property, e_synthesis)

            next_pred = self.model.predict(h_syn[..., -1:, :]).flatten()  # (num_samples * num_conditions, V)
            logp = next_pred.logp.view(self.num_samples, -1, next_pred.logp.shape[-1])
            logp = (logp * weight[None, :, None]).sum(dim=1)  # (num_samples, V)
            next_tokens = torch.multinomial(logp.softmax(dim=-1), num_samples=1).squeeze(-1)

            builder.append(**next_pred.unflatten_tokens(next_tokens))
            if builder.ended.all():
                break

        return builder.get()

    @torch.no_grad()
    def sample(self, query: Query) -> SynthesisRepr:
        planner = QueryPlanner(query)
        property_repr_list = planner.get_property_reprs()
        weight_list = planner.get_weights()

        sample_list: list[SynthesisRepr] = []
        for prop_repr, weight in zip(property_repr_list, weight_list):
            samples = self._sample_conjunctive(prop_repr, weight)
            sample_list.append(samples)

        return concat_synthesis_reprs(*sample_list)
