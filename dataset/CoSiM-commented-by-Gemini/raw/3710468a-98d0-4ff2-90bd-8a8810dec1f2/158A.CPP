/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" contest problem.
 *
 * This program determines how many contestants will advance to the next round
 * of a competition. The rule is that a contestant advances if their score is
 * positive and is greater than or equal to the score of the k-th place finisher.
 * The input scores are guaranteed to be in non-increasing order.
 *
 * Algorithm:
 * 1. Read the number of contestants (n) and the rank of the cutoff score (k).
 * 2. Read all 'n' scores into an array.
 * 3. Identify the cutoff score, which is the score of the k-th person (at index k-1).
 * 4. Iterate through all contestants' scores.
 * 5. Count how many have a score that is both positive and meets or exceeds the
 *    cutoff score.
 * 6. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of contestants, due to the
 * two separate passes for reading and checking scores.
 * Space Complexity: O(N) to store the scores in an array.
 */
#include <iostream>
using namespace std;
int main() {
  // n: number of contestants.
  // k: the rank that determines the cutoff score.
  // a: array to store the scores.
  // c: counter for contestants who advance.
  int n, k, a[50], c = 0;
  
  cin >> n >> k;
  
  // Read all n scores into the array.
  for (int i = 0; i < n; i++)
    cin >> a[i];

  /**
   * Block Logic: Count the number of participants who advance.
   * Iterates through each participant's score.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Advancement Condition Check.
     * A participant advances if their score a[i] is positive AND
     * is greater than or equal to the score of the k-th participant (a[k-1]).
     */
    if (a[i] >= a[k - 1] && a[i] > 0)
      c++;
  }
  
  // Print the total count of advancing participants.
  cout << c;
  return 0;
}
