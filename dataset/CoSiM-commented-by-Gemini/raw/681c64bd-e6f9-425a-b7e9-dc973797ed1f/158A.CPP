#include <bits/stdc++.h>

using namespace std;

/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem from Codeforces (Problem 158A).
 * @details This program determines the number of contestants who advance to the next round.
 * A contestant advances if their score is positive and greater than or equal to the
 * k-th place finisher's score.
 *
 * Algorithm:
 * This solution identifies the number of advancing contestants by finding the rank of the
 * last qualifying participant. The input scores are in non-increasing order.
 * 1. It starts by considering the k-th ranked person (at index `k-1`) as the potential last
 *    qualifier. The index is stored in `r`.
 * 2. It extends `r` to include all contestants who are tied with the k-th person, since
 *    they all have the same score and rank.
 * 3. It then retracts `r`, moving backwards from the last tied person, to exclude anyone
 *    with a score of 0, as a positive score is required to advance.
 * 4. The final value of `r` is the 0-based index of the last person to qualify. The total
 *    number of qualifiers is therefore `r + 1`.
 *
 * Time Complexity: O(n), where `n` is the number of participants. The loops for reading
 * and checking scores are linear.
 * Space Complexity: O(n) to store the scores in a vector.
 */
int main() {

  int n, k;
  cin >> n >> k;

  vector<int> v;

  for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    v.push_back(x);
  }

  // Start with the initial rank `k` (0-indexed `k-1`) as our cutoff.
  int r = k - 1;

  /**
   * @brief Extend the cutoff to include all contestants tied with the k-th place score.
   * Since scores are non-increasing, we just check forward from rank `r`.
   */
  while (r < n - 1 && v[r + 1] == v[k - 1])
    r++;

  /**
   * @brief Retract the cutoff to exclude contestants with a score of 0.
   * A positive score is required. We check backwards from `r`. If the last
   * person `r` has a score of 0, they (and anyone before with a 0) don't qualify.
   */
  while (r >= 0 && v[r] == 0)
    r--;

  // The number of qualified contestants is the 0-based index of the last
  // qualifier plus one.
  cout << r + 1 << endl;

  return 0;
}
