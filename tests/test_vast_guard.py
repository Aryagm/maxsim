import json

from ops.vast.vast_guard import InstanceRecord, destroyable_owned_instances, projected_spend


def test_destroyable_owned_instances_only_returns_ledger_ids_with_matching_live_labels():
    ledger = [
        InstanceRecord(instance_id=101, label="maxsim-v0-20260630-abcd-cuda-smoke", offer_id=1, hourly_cost=0.50),
        InstanceRecord(instance_id=202, label="maxsim-v0-20260630-abcd-large", offer_id=2, hourly_cost=1.25),
    ]
    live_instances = [
        {"id": 101, "label": "maxsim-v0-20260630-abcd-cuda-smoke"},
        {"id": 202, "label": "renamed-by-user"},
        {"id": 303, "label": "maxsim-v0-foreign"},
    ]

    assert destroyable_owned_instances(ledger, live_instances) == [101]


def test_projected_spend_uses_existing_ledger_plus_new_instance_runtime():
    ledger = [
        InstanceRecord(instance_id=101, label="maxsim-v0-a", offer_id=1, hourly_cost=0.50),
        InstanceRecord(instance_id=202, label="maxsim-v0-b", offer_id=2, hourly_cost=1.25),
    ]

    spend = projected_spend(ledger, new_hourly_cost=2.0, planned_hours=3.0)

    assert spend == 11.25


def test_instance_record_round_trips_json():
    record = InstanceRecord(instance_id=101, label="maxsim-v0-a", offer_id=55, hourly_cost=0.75)

    encoded = json.dumps(record.to_json())
    decoded = InstanceRecord.from_json(json.loads(encoded))

    assert decoded == record

