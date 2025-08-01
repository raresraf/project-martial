import uuid

def generate_uids(count):
    """
    Generates a specified number of UUIDs.

    Args:
        count (int): The number of UUIDs to generate.

    Returns:
        list: A list of generated UUID strings.
    """
    uids = []
    for _ in range(count):
        uids.append(str(uuid.uuid4()))
    return uids

num_uids_to_generate = 200
generated_uids = generate_uids(num_uids_to_generate)

print(f"Generated {num_uids_to_generate} UIDs:")
for i, uid in enumerate(generated_uids):
    print(f"\"{uid}\",")

