/**
 * @5ecde271-7602-4e6b-bfd4-736f95c37c8b/158A.CPP
 * @brief This program solves problem 158A, which involves determining the number
 * of participants who will advance to the next round of a competition.
 * Qualification criteria: a participant's score must be strictly positive
 * AND greater than or equal to the score of the participant at the k-th position.
 * Scores are assumed to be provided in non-increasing order.
 *
 * Algorithm: Two-pass linear scan. First pass reads scores. Second pass counts
 * participants meeting the criteria using the k-th score as a threshold.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(1) as the array size is fixed (100).
 */

#include <iostream> // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

int main() {
  int n, k; // Declares 'n' (total participants) and 'k' (target rank).
  cin >> n; // Reads 'n' from standard input.
  cin >> k; // Reads 'k' from standard input.
  int a[100]; // Declares a fixed-size array 'a' to store participant scores (assuming N <= 100).
  
  /**
   * Block Logic: First pass - reads 'n' participant scores into the 'a' array.
   * Invariant: After this loop, 'a' contains all 'n' participant scores.
   * Pre-condition: Scores are typically provided in non-increasing order.
   */
  for (int i = 0; i < n; i++) { // Loops through each participant.
    cin >> a[i]; // Reads each score.
  }
  
  int c = 0; // Initializes 'c' to 0, which will count the qualifying participants.
  /**
   * Block Logic: Second pass - iterates through all scores to count qualifying participants.
   * A participant qualifies if their score is non-zero AND greater than or equal to
   * the score of the participant at the (k-1)th index (k-th place).
   * Invariant: 'c' accurately reflects the number of participants meeting the criteria.
   */
  for (int i = 0; i < n; i++) { // Loops through each participant.
    // Inline: Checks if the current participant's score is greater than or equal to the k-th score
    // AND if their score is strictly positive.
    if (a[i] >= a[k - 1] && a[i] != 0) { 
      c++; // Increments 'c' if both conditions are met.
    }
  }
  cout << c; // Prints the final count of qualifying participants.
  // The program implicitly returns 0 on successful completion of main.
}
