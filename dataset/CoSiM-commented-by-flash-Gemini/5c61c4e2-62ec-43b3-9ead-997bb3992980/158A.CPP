/**
 * @5c61c4e2-62ec-43b3-9ead-997bb3992980/158A.CPP
 * @brief This program solves problem 158A, which involves determining the number
 * of participants who will advance to the next round of a competition.
 * Qualification criteria: a participant's score must be strictly positive
 * AND greater than or equal to the score of the participant at the k-th position.
 * Scores are assumed to be provided in non-increasing order.
 *
 * Algorithm: Two-pass linear scan. First pass reads scores. Second pass counts
 * participants meeting the criteria using the k-th score as a threshold.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) for storing participant scores.
 */

#include <iostream> // For standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace.

int main(int argc, const char *argv[]) {
  int n, k, count; // Declares 'n' (total participants), 'k' (target rank), 'count' (qualifiers).
  cin >> n;        // Reads 'n' from standard input.
  cin >> k;        // Reads 'k' from standard input.

  int scores[n]; // Declares an array 'scores' of size 'n' to store participant scores.
  /**
   * Block Logic: First pass - reads 'n' participant scores into the 'scores' array.
   * Invariant: After this loop, 'scores' array contains all participant scores.
   * Pre-condition: Scores are typically provided in non-increasing order.
   */
  for (int i = 0; i < n; i++) { // Loops through each participant.
    cin >> scores[i]; // Reads each score.
  }

  count = 0; // Initializes 'count' of qualifying participants to 0.
  /**
   * Block Logic: Second pass - iterates through all scores to count qualifying participants.
   * A participant qualifies if their score is non-zero AND greater than or equal to
   * the score of the participant at the (k-1)th index (k-th place).
   * Invariant: 'count' accurately reflects the number of participants meeting the criteria.
   */
  for (int i = 0; i < n; i++) { // Loops through each participant.
    // Inline: Checks if the current participant's score is greater than or equal to the k-th score
    // AND if their score is strictly positive.
    if (scores[i] >= scores[k - 1] && scores[i] != 0) 
      count++; // Increments 'count' if both conditions are met.
  }
  cout << count; // Prints the final count of qualifying participants.

  return 0; // Indicates successful program termination.
}
