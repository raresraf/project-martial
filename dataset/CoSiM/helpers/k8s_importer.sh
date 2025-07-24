#!/bin/bash

# Define the repository URL
REPO_URL="https://github.com/kubernetes/kubernetes"

# Define the base path for your folders
BASE_FOLDER_PATH="/Users/raf/code/project-martial/dataset/CoSiM/raw"

# Define your lists of commit SHAs and UIDs
# Commit SHAs - replace with your actual list of SHAs
COMMIT_SHAS=(
  # "13f0449e4cace8bd3a6dd1f0da642f0713433707"
  # "36f5820ad156c2353149b1cfd20f38d7be6a0943"
  # "b5e9a8262e4a45393d13dfc2ba01e810c6c45dfb"
  # "303593cafe1163f0813948f42c9e0f7561017c8f"
  # "f001b3916d4775b75910c6ef65578cfeb4fa90a7"
  # "b62503dd66ecee08022055a6c1b7c657440f5130"
  # "6d84838c0f92b3b1c7d417d782ab1b534dc01a63"
  # "b7089f81da573c6e33c27171adcd6991eaf125e9"
  # "0e485310cf16eb72864ada400b1849a35642c33d"
  # "48007fc32bb473d1d0d7d7c24b9daa2157e433d1"
  # "d7e4028437a4ab75f103471fdf911b503834af80"
  # "c9f5eea1f6146813ee0f3e2ec485d0ad3d329670"
  # "61de4fc34df5299a7946b61a825cd246cc540ba5"
  # "cca3d557e6ff7f265eca8517d7c4fa719077c8d1"
  # "318b089918e8aa5fe0f98b8d2b754417b36e1f7c"
  # "ba690c2257af76bd971d0dfb6bef13ff9099e549"
  # "8d3973c2f4ee9a24e696aaa1491d1394df33d8cd"
  # "0deec63d1a28cba1b1c409fd90b02fcfca376018"
  # "b1ebeffeb37ddcb168421c97d1e4f6f9649885d1"
  # "554ff48da20978765641973f2c838c86fb1e0cd7"
  # "dcc302ab382288a2101bea2e65b469fcde2323e1"
  # "2425bfe265acdf4caa225a7c715b48af7e895626"
  # "096da24c6ecfecb6ce4729ee8b60ea7598d4af38"
  # "7e71c49bbf538d92b3cb28dc61ee04dace24bd34"
  # "d150362e63471ed8e76d7420c3d61b8a233a368e"
)

# UIDs - ensure this list has at least twice the number of entries as COMMIT_SHAS
# (one for old_folder_name and one for new_folder_name per iteration)
UIDS=(
    "d4669442-ef53-4045-9860-70ca3f232a2c"
    "e7bdf167-6519-4b64-9baa-923fedc42c7c"
    "d27d3478-8538-4583-b41d-371a37249166"
    "2db59c5b-7ba6-46f3-a9e7-7b74dae933dc"
    "dd72a4a2-9707-4866-827f-577903da370f"
    "6dbf40b6-a41e-42cf-8104-47702071fdc2"
    "bc137b78-969d-4aad-93f0-2c462a7f153a"
    "e29b3d43-d932-45ca-85b6-9a1c05d76bbd"
    "fbbec3c5-122c-4e17-b756-1b8dbe4de67b"
    "50a83b6f-0188-459f-8c32-a933162427b1"
    "fe313be8-c73f-4688-a698-a75a2cb03fbb"
    "6aef60bf-f562-49a7-b69f-0e1708b82954"
    "fc000201-9b17-46c7-9bf5-47f182e9376b"
    "6332c0a0-bb55-441c-8d1a-d0683a284eac"
    "acdebc03-1db0-4307-aa02-05e6c8217621"
    "1aaafb30-1e01-4fa9-839a-22723af1b584"
    "bfe66009-f073-4ba7-8fec-70e84598b187"
    "e8d4e542-fad4-4cec-800a-24df04d1a65c"
    "dca958c5-f65b-4c9b-827c-0edcd873a77e"
    "acf7371d-e13f-4e57-a517-2c46d5443fd6"
    "3ad9ef5e-8a3a-4fe5-853d-d91c0001d199"
    "a56e30a8-4183-499b-b929-758a2ed37ae4"
    "5080340b-a11d-401b-9440-225841c5d384"
    "bf772a8c-f945-4675-b797-924aaba377e7"
    "80b25dee-01d0-45bf-a705-538101285ffd"
    "bb7a6ad6-4cae-40fd-ae5a-4a0496513e65"
    "3e8b09be-b47f-4a8a-8a22-100595072f82"
    "75f4b4c4-f693-418a-ba42-a22a5e0ac2ea"
    "b42d16e7-bcad-4b98-92f6-58038b0aec35"
    "0b950450-5f04-439f-807c-c4d5e79b6068"
    "2050bcfd-f3e8-4851-8bdd-25c0651f8839"
    "b5c0892d-2d5f-469e-a347-ebd2633a87ff"
    "02b4bd14-2748-4e9a-9c0d-c1977f39308c"
    "8422fd48-bc0f-487a-8a08-6cf1748f234c"
    "2ddbe1b5-b12f-4fbf-8c34-3cc81f66a8f7"
    "3d4bc0b2-4449-4b95-9c58-7280095cd64d"
    "8fb566f8-f8ae-44a0-9947-699ca8a99f83"
    "f8e7b130-a4be-48e8-8cd1-142566be21f4"
    "d03e9cf6-e4d8-4e34-8b74-46dae92cddcc"
    "7287eecc-9ea2-4bd0-b64b-8273829e4312"
    "a4d338a9-8de5-44e7-8c38-d763d86ad4d6"
    "3f7cd93e-15b9-46ad-a02b-90045fb69d1f"
    "8bbae96b-aefb-4569-ba24-72503aaec98c"
    "569c5b68-9983-44e4-b2fb-7afa48fdaf15"
    "3969950a-656b-49ba-ba74-27312bceae58"
    "8d32982c-cc61-44b1-9e23-910d6c1a5732"
    "dfc4118c-3359-463b-b20c-8d2cf0ad5c50"
    "462039cb-dc87-4942-8016-85543e52d964"
    "3711c8a9-5b8e-4511-aea8-b780b2b19c2a"
    "cf6bf76d-387e-4722-b8a8-b870b1cb80f3"
)

# Check if there are enough UIDs for the operations
if [ ${#UIDS[@]} -lt $(( ${#COMMIT_SHAS[@]} * 2 )) ]; then
  echo "Error: Not enough UIDs provided for all commit SHAs."
  echo "You need at least $(( ${#COMMIT_SHAS[@]} * 2 )) UIDs, but only ${#UIDS[@]} were provided."
  exit 1
fi

# Loop through each commit SHA
for i in "${!COMMIT_SHAS[@]}"; do
  COMMIT_SHA="${COMMIT_SHAS[$i]}"
  
  # Calculate the indices for UIDs
  OLD_UID_INDEX=$(( i * 2 ))
  NEW_UID_INDEX=$(( i * 2 + 1 ))

  # Get the UIDs for the current iteration
  OLD_FOLDER_UID="${UIDS[$OLD_UID_INDEX]}"
  NEW_FOLDER_UID="${UIDS[$NEW_UID_INDEX]}"

  # Construct the full folder paths
  OLD_FOLDER_NAME="${BASE_FOLDER_PATH}/${OLD_FOLDER_UID}"
  NEW_FOLDER_NAME="${BASE_FOLDER_PATH}/${NEW_FOLDER_UID}"

  echo "Running with:"
  echo "  Commit SHA: ${COMMIT_SHA}"
  echo "  Old Folder: ${OLD_FOLDER_NAME}"
  echo "  New Folder: ${NEW_FOLDER_NAME}"
  echo "---"

  # Execute the python script
  python3 github_importer.py \
    --repo_url="${REPO_URL}" \
    --commit_sha="${COMMIT_SHA}" \
    --old_folder_name="${OLD_FOLDER_NAME}" \
    --new_folder_name="${NEW_FOLDER_NAME}"
done