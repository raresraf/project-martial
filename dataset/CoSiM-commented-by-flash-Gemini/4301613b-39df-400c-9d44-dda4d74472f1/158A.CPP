/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 158A - "Next Round").
 *
 * This program calculates the number of participants who advance to the next round
 * based on their scores relative to the k-th place finisher and a non-zero score requirement.
 * This specific implementation uses a two-phase linear scan with early termination
 * conditions that might differ from a typical interpretation of the problem statement.
 *
 * Algorithm: Two-phase linear scan with early termination.
 * Phase 1: Reads the first `k` scores. If a zero score is encountered, it implies
 *          no further participants will advance, and the count up to that point is printed.
 *          The `k`-th participant's score is stored.
 * Phase 2: For participants beyond the `k`-th position, it compares their score
 *          with the `k`-th score. If a score is found that is *not equal* to
 *          the `k`-th score, the current count is printed, and the program terminates.
 *          If all `n` scores are processed without early termination, `n` is printed.
 *
 * Time Complexity: O(N) in the worst case, where N is the total number of participants.
 * However, due to early termination conditions, it can be faster, potentially O(k) or O(i).
 * Space Complexity: O(1), as only a few integer variables are used.
 */
#include <stdio.h> // Standard input/output library for C-style I/O.

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads participant data, applies specific advancement rules, and prints the count
 * of advancing participants with potential early exits.
 */
int main() {
  int n, k; // 'n' for total participants, 'k' for the k-th place threshold.

  // Pre-condition: Read the total number of participants 'n' and the k-th position 'k'.
  scanf("%d%d", &n, &k);

  int a; // Variable to temporarily store scores read and to eventually hold the k-th score.

  // Block Logic: First phase - process the first 'k' participants.
  // Invariant: 'i' counts the participants from 0 to k-1.
  for (int i = 0; i < k; i++) {
    // Pre-condition: Read the current participant's score.
    scanf("%d", &a);
    // Functional Utility: Early termination condition - if a score is 0,
    // all participants up to this point (excluding this zero-scorer) are counted.
    if (a == 0) {
      printf("%d", i); // Print the count of participants who passed.
      return 0;        // Exit the program.
    }
  }
  
  // Note: At this point, 'a' holds the score of the k-th participant.

  int b; // Variable to temporarily store scores of participants after the k-th one.
  // Block Logic: Second phase - process participants from k-th position up to n-1.
  // Invariant: 'i' counts participants starting from the k-th position.
  for (int i = k; i < n; i++) {
    // Pre-condition: Read the current participant's score.
    scanf("%d", &b);
    // Functional Utility: Early termination condition - if a score is *not equal*
    // to the k-th score ('a'), implies a specific termination condition.
    // Note: This '!= a' condition might differ from a typical interpretation
    // of "score equal to or greater than k-th score" if scores can be greater than 'a'.
    if (b != a) {
      printf("%d", i); // Print the count of participants who passed.
      return 0;        // Exit the program.
    }
  }

  // If the program reaches here, it means all 'n' participants satisfy the criteria.
  printf("%d", n); // Print the total number of participants.
  return 0;
}
