/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem.
 *
 * The problem is to find the minimum number of coins to take from a set
 * of coins of varying values, such that the sum of the values of the coins
 * taken is strictly greater than the sum of the values of the remaining coins.
 *
 * @algorithm Greedy Approach
 * @details The optimal strategy is to always pick the coins with the largest values first.
 * This greedy approach ensures that the sum grows as quickly as possible, thus
 * minimizing the number of coins needed.
 * The program calculates the total sum of all coin values, then sorts the coins in
 * descending order. It iteratively takes the largest coins and adds their values
 * to a running total, until this total exceeds half of the overall sum.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting of the coins.
 * Space Complexity: O(N) to store the values of the `N` coins in a vector.
 */

#include <algorithm>
#include <iostream>
#include <vector>
#define optimizar_io                                                           
  ios_base::sync_with_stdio(0);                                                
  cin.tie(0);
using namespace std;
int main() {
  optimizar_io;
  long long a, sum = 0, sum1 = 0, cont = 0;
  cin >> a;
  vector<long long> v(a);

  // Read all coin values and calculate their total sum.
  for (int i = 0; i < a; ++i) {
    cin >> v[i];
    sum += v[i];
  }

  // The target is to have a sum strictly greater than the sum of the rest.
  // This is equivalent to having a sum greater than half of the total sum.
  sum /= 2;

  // Sort the coins in ascending order to easily access the largest ones.
  sort(v.begin(), v.end());

  /**
   * @brief Greedily take the largest coins.
   * Iterate from the largest coin downwards, adding its value to `sum1`.
   * Count the number of coins taken. Stop when `sum1` is greater than `sum`.
   */
  for (int i = a - 1; i >= 0; --i) {
    sum1 += v[i];
    cont++;
    if (sum1 > sum)
      break;
  }

  cout << cont << "
";
  return 0;
}
