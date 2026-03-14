/**
 * @67e047f5-bb64-4199-95e7-c8c1198ddbdc/158A.CPP
 * @brief This program solves Codeforces problem 158A, "Next Round".
 * It determines the number of participants who will advance to the next round
 * of a competition. Qualification criteria: a participant must score equal to
 * or greater than the k-th place finisher's score, and must have a positive score.
 *
 * Algorithm: Linear scan. Reads all scores, then iterates through them to count
 * those meeting the qualification criteria against the k-th score.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) for storing scores in an array.
 */
#include <iostream> // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to determine qualifying participants based on scores.
 * Reads number of participants and k-th place, then all scores, and counts qualifiers.
 * @return 0 on successful execution.
 */
int main() {
  int n, k; // Declares 'n' for the number of participants, and 'k' for the k-th place.
  cin >> n >> k; // Reads 'n' and 'k' from standard input.

  int scores[n]; // Declares an array to store the scores of 'n' participants.
  
  /**
   * Block Logic: Reads 'n' scores from standard input into the `scores` array.
   * Invariant: After this loop, `scores` contains all participant scores.
   */
  for (int i = 0; i < n; i++) // Loops 'n' times, once for each participant.
    cin >> scores[i]; // Reads each participant's score.

  int count = 0; // Initializes `count` to 0, which will store the number of qualifying participants.
  
  /**
   * Block Logic: Iterates through all participant scores to determine qualifiers.
   * A participant qualifies if their score is greater than 0 and greater than or
   * equal to the score of the k-th place finisher.
   * Pre-condition: `scores` array is populated, and `k-1` is a valid index.
   * Invariant: `count` accumulates the total number of qualifying participants.
   */
  for (int i = 0; i < n; i++) { // Loops through each participant's score.
    // Checks both qualification criteria: positive score AND score >= k-th score.
    if (scores[i] >= scores[k - 1] && scores[i] > 0) 
      count++; // Increments 'count' if both conditions are met.
  }
  cout << count << endl; // Prints the final count of qualifying participants, followed by a newline.

  return 0; // Indicates successful program termination.
}