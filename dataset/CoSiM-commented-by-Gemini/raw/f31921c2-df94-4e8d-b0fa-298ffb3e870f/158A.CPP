/**
 * @file 158A.CPP
 * @brief A unique rank-manipulation solution for the "Next Round" problem.
 *
 * This program determines the number of participants advancing to the next round.
 * A participant advances if they have a positive score and their score is at least
 * as high as the k-th place participant's score. This solution reads all scores
 * into an array and then cleverly manipulates the rank `k` itself to find the
 * final count of qualifiers. It first extends `k` to include all ties at the
 * cutoff rank, and then retracts `k` to exclude any participants with non-positive scores.
 *
 * @author f31921c2-df94-4e8d-b0fa-298ffb3e870f
 *
 * @algorithm Rank Manipulation with Tie and Score Handling
 * @complexity
 * Time: O(N), for reading the N scores into the array. The while loops also
 *       run at most N times in total.
 * Space: O(N), to store the scores in a Variable Length Array (VLA).
 */
// Codeforces_Andrespengineer
/* List of solved problems
 *
 * Andr s Sierra
 * Instituto Tecnol gico de Santo Domingo (INTEC)
 *
 */

#include <cstdlib>
#include <iostream>
using namespace std;

int main() {

  // `n`: total participants, `k`: cutoff rank, `p`: temporary score holder.
  int n, k, p;
  
  cin >> n >> k;

  // `arr`: a VLA to store all participant scores.
  int arr[n];

  // Read all `n` scores into the array.
  for (int i = 0; i < n; i++) {
    cin >> p;
    arr[i] = p;
  }

  // Adjust k to be a 0-based index.
  k = k - 1;

  /**
   * @brief This loop extends the cutoff rank `k` to include all ties.
   * If the person at rank `k` has the same score as the person at `k+1`,
   * the rank `k` is effectively extended to include the next person.
   * This continues until a person with a lower score is found or the end of the list is reached.
   */
  while (k + 1 < n) {
    if (arr[k] == arr[k + 1])
      k++;
    else
      break;
  }
  
  /**
   * @brief This loop retracts the cutoff rank `k` to exclude non-positive scores.
   * After finding the full group of qualifiers (including ties), this checks
   * if the last person has a score of 0. If so, they are disqualified, and the
   * rank is moved back until a qualifier with a positive score is found.
   */
  while (k >= 0) {
    if (arr[k] == 0)
      k--;
    else
      break;
  }

  /**
   * @brief Print the final count.
   * After the manipulations, `k` is the 0-indexed rank of the last person
   * who qualifies. Therefore, `k + 1` gives the total number of qualifiers.
   */
  cout << k + 1 << endl;
}