/**
 * @file 158A.CPP
 * @brief Solution to a problem that counts participants advancing in a competition.
 *
 * This program determines how many participants in a competition advance to the next round.
 * Qualification is based on two criteria: a participant's score must be strictly positive,
 * and it must be greater than or equal to the score of the k-th placed participant.
 * The program reads the total number of participants 'n', the k-th position 'k',
 * and then all 'n' scores. It then calculates and prints the count of qualified participants.
 *
 * Algorithm: Reads all scores, identifies the score threshold based on the k-th participant,
 *            and then iterates through all scores again to count those meeting the criteria.
 * Time Complexity: O(N), where N is the number of participants, due to two linear passes
 *                  over the participant scores.
 * Space Complexity: O(N) for storing the participant scores in an array.
 */
#include <bits/stdc++.h> // Includes various standard libraries, common in competitive programming.
using namespace std;     // Using the standard namespace.

int main() {
  // Declare 'n' for total participants, 'k' for the k-th place, and 'c' (count) initialized to 0.
  int n, k, c = 0;
  // Read the total number of participants 'n' and the k-th place 'k'.
  cin >> n >> k;
  // Declare a Variable Length Array (VLA) 'a' of size 'n' to store participant scores.
  // This is a C99 feature and a GNU extension in C++, often used in competitive programming.
  int a[n];
  /**
   * Block Logic: Reads 'n' participant scores from input into the array 'a'.
   * Invariant: 'i' increments from 0 to 'n-1', filling each element of the array.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i]; // Read the score for the i-th participant.
  }
  /**
   * Block Logic: Iterates through the scores to count participants who qualify.
   * Qualification criteria: score is positive AND score is >= k-th participant's score.
   * Invariant: 'i' increments from 0 to 'n-1', checking each participant's score.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Checks if the current participant meets the two qualification conditions.
     * Pre-condition: 'a[i]' is the current participant's score, 'a[k-1]' is the k-th participant's score.
     */
    if (a[i] >= a[k - 1] && a[i] > 0)
      // If both conditions are met, increment the counter 'c'.
      c++;
  }
  // Output the total count of qualified participants.
  cout << c;
  return 0; // Indicate successful program execution.
}
