/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem (Codeforces 158A).
 *
 * This program determines the number of participants who advance to the next round of a competition.
 * Participants advance if their score is positive and is greater than or equal to the score
 * of the participant in the k-th place.
 * Algorithm: Read all participant scores. Then, iterate through the scores, comparing each
 *            with the k-th score (which is a[k-1] in 0-indexed array) and checking for positivity.
 * Time Complexity: O(N) where N is the number of participants, due to two passes (reading and counting).
 * Space Complexity: O(N) to store participant scores.
 */

#include <iostream> // Required for standard input/output operations (std::cin, std::cout).

// Global variables to store input parameters and intermediate results.
// n: total number of participants.
// k: the k-th place threshold.
// i, j: loop counters/indices.
// a[51]: array to store scores of up to 50 participants (plus one for 0-indexing safety or larger k).
int n, k, i, j, a[51];

/**
 * @brief Main function to calculate the number of advancing participants.
 *
 * Reads competition parameters and participant scores, then counts and prints
 * the number of participants who meet the advancement criteria.
 * @return Implicitly 0 on successful execution as main() with no explicit return defaults to 0.
 */
int main() {
  // Block Logic: Read the total number of participants 'n' and the k-th place threshold 'k'.
  // Pre-condition: User provides two integers.
  // Invariant: 'n' and 'k' hold the respective input values.
  std::cin >> n >> k;

  // Block Logic: Read 'n' participant scores into the array 'a'.
  // The loop runs 'n' times, incrementing 'i' to store scores at successive indices.
  // Pre-condition: 'n' is within the bounds of array 'a'.
  // Invariant: 'a' contains the scores of all 'n' participants.
  while (n > i)
    std::cin >> a[i++];

  // Block Logic: Count participants who advance to the next round.
  // A participant advances if their score is positive and at least equal to the k-th score.
  // The loop iterates while the current score 'a[j]' is positive and
  // greater than or equal to the k-th score (a[k-1]).
  // Pre-condition: Array 'a' is populated with scores, 'k-1' is a valid index.
  // Invariant: 'j' counts the number of participants meeting the criteria.
  while (a[j] and a[j] >= a[k - 1])
    ++j;
  
  // Block Logic: Print the final count of advancing participants.
  // Post-condition: The number of participants who advance is printed to standard output.
  std::cout << j;
}
