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
        # uuid.uuid4() generates a random UUID
        uids.append(str(uuid.uuid4()))
    return uids

# Generate 100 UIDs
num_uids_to_generate = 100
generated_uids = generate_uids(num_uids_to_generate)

# Print the generated UIDs
print(f"Generated {num_uids_to_generate} UIDs:")
for i, uid in enumerate(generated_uids):
    print(f"{i+1}: {uid}")

