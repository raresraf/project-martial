/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (160A) from Codeforces.
 * @details The goal is to find the minimum number of coins to take such that their sum
 * is strictly greater than the sum of the remaining coins.
 *
 * @author Abhishek Gupta | TCET,Mumbai | BE.IT (as per original comment)
 *
 * @algorithm This solution uses a greedy strategy. It sorts the coins in descending order
 * and takes the largest ones until their sum exceeds half of the total sum.
 * 1. Read the number of coins (n) and their values, calculating the total sum `s`.
 * 2. Sort the coins in descending order to pick the largest ones first.
 * 3. Divide the total sum `s` by 2. The goal is to accumulate a sum `s1` strictly greater than this value.
 *    (This is equivalent to `s1 > s_total - s1`).
 * 4. Iterate through the sorted coins, adding their values to `s1` and counting the number of coins taken (`c`).
 * 5. As soon as `s1` exceeds `s/2`, the minimum number of coins has been found. Break the loop.
 * 6. Print the count `c`.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N), for storing the coin values in an array.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, i, x; // The variable `x` is declared but never used.
  cin >> n;
  int s = 0, s1 = 0, c = 0;
  // A variable-length array (VLA) is a non-standard C++ extension.
  int a[n];
  /**
   * @block
   * @brief Reads coin values and calculates the total sum.
   */
  for (i = 0; i < n; i++) {
    cin >> a[i];
    s += a[i];
  }
  // Sort coins in descending order to greedily pick the largest ones.
  sort(a, a + n, greater<int>());

  // The target is to have a sum strictly greater than the sum of the rest.
  // This is equivalent to having a sum greater than half of the total.
  s /= 2;
  /**
   * @block
   * @brief Greedily takes coins until their sum exceeds half the total sum.
   * @invariant `s1` holds the sum of the `c` largest coins.
   * @pre `a` is sorted descending, `s` is half the total sum.
   * @post `c` holds the minimum number of coins needed.
   */
  for (i = 0; i < n; i++) {
    s1 += a[i];
    c++;
    if (s1 > s) {
      break;
    }
  }
  cout << c;
  return 0;
}
