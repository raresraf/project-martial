/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem (Codeforces 158A).
 * @author competitive-programmer
 *
 * @details This program calculates the number of participants who will advance
 * to the next round of a competition. The rule for advancing is that a
 * participant's score must be equal to or greater than the score of the k-th
 * place finisher, and the score must be positive. The input list of scores
 * is guaranteed to be in non-increasing order.
 *
 * Algorithm:
 * 1. Read the total number of participants (n) and the rank (k) to compare against.
 * 2. Read all 'n' scores into a vector.
 * 3. The score to beat is the score of the k-th participant, which is at index `k-1`.
 * 4. Iterate through the participants' scores. For each participant, check if their
 *    score is greater than or equal to the k-th place score AND if their score
 *    is positive.
 * 5. If both conditions are met, increment a counter.
 * 6. Since the scores are in non-increasing order, if a participant does not meet
 *    the criteria, no subsequent participant will either. The loop can be terminated
 *    early using a `break` statement.
 * 7. Print the final count.
 *
 * Time Complexity: O(n) in the worst case to read the input, but the counting
 * loop may terminate early, making it O(k) in many cases.
 * Space Complexity: O(n) to store the scores in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Optimizes C++ standard streams for faster I/O.
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  // n: number of participants, k: the rank to compare against.
  int n, k, temp, cnt = 0;
  cin >> n >> k;

  // Read all 'n' participant scores into a vector.
  vector<int> vec(n);
  for (int i = 0; i < n; i++)
    cin >> vec[i];

  /**
   * Block Logic: Iterate through the participants to count who advances.
   * The score to qualify is `vec[k-1]`.
   */
  for (int i = 0; i < n; i++)
    /**
     * Advancement Condition: The participant's score `vec[i]` must be
     * greater than or equal to the k-th participant's score `vec[k-1]`,
     * AND the score `vec[i]` must be positive.
     */
    if (vec[k - 1] <= vec[i] && vec[i])
      cnt++;
    else
      /**
       * Optimization: Since the scores are in non-increasing order, the first
       * time a participant fails to qualify, no one after them can qualify either.
       * The loop can be safely terminated.
       */
      break;

  // Print the total count of participants who advance.
  cout << cnt << "\n";
}
