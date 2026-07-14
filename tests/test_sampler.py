from rc_irstd.data.sampler import DomainBalancedBatchSampler


def test_balanced_sampler_yields_a_batch_when_per_domain_exceeds_domain_size() -> None:
    sampler = DomainBalancedBatchSampler(
        domain_ids=[0, 1],
        batch_size=8,
        shuffle=False,
        drop_last=True,
        seed=0,
    )
    batches = list(sampler)
    assert len(batches) == 1
    assert len(batches[0]) == 8
