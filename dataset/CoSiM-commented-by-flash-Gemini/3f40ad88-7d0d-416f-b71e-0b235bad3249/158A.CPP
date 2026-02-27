/**
 * @file 158A.CPP
 * @brief Solution for competitive programming problem: "Next Round".
 *
 * @details This program calculates the number of participants who will advance
 *          to the next round of a competition. Participants advance if their
 *          score is greater than or equal to the k-th place finisher's score,
 *          provided their score is strictly positive. This implementation uses
 *          C-style input/output and a variable-length array to store scores.
 *          Algorithm: Two-pass iterative processing of scores.
 *          Time Complexity: O(N), where N is the number of participants, as scores
 *                           are read and processed in two separate linear passes.
 *          Domain: Competitive Programming.
 */
#include <cstdio>

using namespace std;

int main() {
  // Declare variables:
  // 'N': total number of participants.
  // 'k': the 1-based index of the k-th place participant.
  int N, k;
  // Read N and k from standard input using `fscanf`.
  fscanf(stdin, "%d%d", &N, &k);
  
  // Declare a variable-length array (VLA) 'A' to store participant scores.
  // Note: VLAs are a C99 feature and a GCC extension in C++, not standard C++.
  int A[N];
  int i; // Loop counter.

  // Block Logic: Read all participant scores into the array 'A'.
  for (i = 0; i < N; ++i) {
    fscanf(stdin, "%d", A + i); // Read score directly into the array element's address.
  }
  
  // Get the score of the k-th place participant (0-indexed: A[k-1]).
  int num = A[k - 1];
  int count = 0; // Counter for participants who advance.
  
  // Block Logic: Iterate through the scores to count participants who meet the advancement criteria.
  for (i = 0; i < N; ++i) {
    // Precondition: A participant advances if their score is greater than or equal to
    //               the k-th place score (`num`) AND their score is strictly positive.
    if (A[i] >= num && A[i] > 0) {
      ++count; // Increment the counter for advancing participants.
    }
  }
  // Print the final count of participants who advance to standard output using `fprintf`.
  fprintf(stdout, "%d", count);

  return 0; // Indicate successful program termination.
}
