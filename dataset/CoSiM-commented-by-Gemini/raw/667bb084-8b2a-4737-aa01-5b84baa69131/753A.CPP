#include <iostream>
#include <vector>
using namespace std;

/**
 * @file 753A.CPP
 * @brief Solution to the "Santa Claus and Candies" problem (Codeforces 753A).
 * @details This program partitions a given number of candies, N, among the maximum
 * possible number of children, such that each child receives a distinct number of candies.
 *
 * Algorithm:
 * A greedy approach is used. The program iterates through integers i = 1, 2, 3, ... and
 * distributes `i` candies to the i-th child as long as the remaining number of candies `N`
 * is sufficient. When `N` is less than the next number of candies to be distributed (`i`),
 * the loop stops. To ensure all candies are distributed, the remaining `N` candies are given
 * to the last child who received candies. This preserves the distinctness of the candy amounts
 * and maximizes the number of children.
 *
 * Time Complexity: O(sqrt(N)). The number of children `k` is such that the sum 1 + 2 + ... + k
 * is approximately N. This sum is k*(k+1)/2, so k is proportional to sqrt(N).
 * Space Complexity: O(sqrt(N)) to store the number of candies each child receives.
 */
int main() {
  /**
   * @brief Optimizes I/O operations by decoupling C++ streams from C stdio
   * and un-tying cin from cout. This is a common practice in competitive programming
   * for faster input/output.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  int N, orig;
  cin >> N;
  vector<int> taken;
  orig = N;

  /**
   * @brief Greedily assigns a distinct number of candies to each child.
   * Pre-condition: N holds the total number of candies.
   * Invariant: At each step `i`, the first `i-1` children have been assigned
   * `1, 2, ..., i-1` candies respectively. `N` holds the remaining candies.
   */
  for (int i = 1; i <= orig; i++) {
    /**
     * @brief Check if the remaining candies are insufficient for the next child.
     * If `N < i`, we cannot give `i` candies. The remaining `N` candies
     * are added to the last child's share to use up all candies.
     */
    if (N < i) {
      taken[taken.size() - 1] += N;
      break;
    }
    N -= i;
    taken.push_back(i);
  }

  // Output the number of children who received candies.
  cout << (int)taken.size() << "\n";
  // Output the number of candies each child received.
  cout << taken[0];
  for (int i = 1; i < (int)taken.size(); i++)
    cout << " " << taken[i];
  cout << "\n";
  return 0;
}
