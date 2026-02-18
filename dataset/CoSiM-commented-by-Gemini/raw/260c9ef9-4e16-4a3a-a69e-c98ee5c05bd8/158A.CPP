/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem (158A) from Codeforces.
 * @details This program determines the number of participants who advance to the next round of a competition.
 * A participant advances if they have a positive score and that score is greater than or equal to the k-th
 * place finisher's score.
 *
 * @algorithm The solution employs a straightforward linear scan approach.
 * 1. Read the total number of participants (n) and the rank of the cutoff score (k).
 * 2. Read all participant scores into an array.
 * 3. Identify the cutoff score, which is the score of the k-th participant (at index k-1).
 * 4. Iterate through the scores again, counting how many are positive and meet or exceed the cutoff score.
 * 5. Print the final count.
 *
 * @complexity
 * Time Complexity: O(N) - The program involves two separate loops that iterate up to N times,
 * one for input and one for counting. This results in a linear time complexity.
 * Space Complexity: O(N) - An array of size N is used to store the scores of the participants.
 */
#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n, k;
  // Read the number of participants and the k-th position.
  cin >> n >> k;
  // A variable-length array (VLA), a non-standard C++ extension, is used to store scores.
  int ar[n], count = 0;
  /**
   * @block
   * @brief Reads all participant scores from standard input.
   * @pre The variable 'n' must be initialized with the number of scores to read.
   * @post The 'ar' array is populated with 'n' scores.
   */
  for (int i = 0; i < n; i++) {
    cin >> ar[i];
  }
  /**
   * @block
   * @brief Counts participants who advance to the next round.
   * @invariant The cutoff_score is ar[k-1]. A participant advances if their score is
   *            positive and >= cutoff_score.
   * @pre The 'ar' array is populated, and 'k' is a 1-based index into the sorted (by rank) list.
   * @post 'count' holds the number of participants who qualify.
   */
  for (int i = 0; i < n; i++) {
    // A participant qualifies if their score is positive and at least as high as the k-th participant's score.
    if (ar[i] >= ar[k - 1] && ar[i] > 0)
      count++;
  }
  // Output the final count of qualifying participants.
  cout << count;
  return 0;
}
