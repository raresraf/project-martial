/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (160A) from Codeforces.
 * @details The goal is to find the minimum number of coins to take such that the sum of their values
 * is strictly greater than the sum of the remaining coins.
 *
 * @algorithm This problem can be solved using a greedy approach.
 * 1. Read all the coin values and calculate their total sum.
 * 2. Sort the coins in non-decreasing order.
 * 3. Iterate through the coins from largest to smallest, accumulating their sum and counting the number of coins taken.
 * 4. In each step, check if the accumulated sum is strictly greater than the sum of the remaining coins.
 * 5. The first time this condition is met, the current count of coins is the minimum required. The loop terminates and this count is printed.
 *
 * @complexity
 * Time Complexity: O(N log N) - The complexity is dominated by the sorting operation on the 'n' coins.
 * Space Complexity: O(N) - An array of size up to 105 (a fixed-size buffer based on problem constraints) is used to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int sum1 = 0, sum2 = 0, cont = 0, diff;
  int n, aa[105]; // Fixed-size array based on problem constraints (n <= 100)
  cin >> n;
  /**
   * @block
   * @brief Reads coin values and calculates the total sum.
   * @pre 'n' is the number of coins to read.
   * @post 'aa' is populated with 'n' coin values and 'sum1' holds their total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> aa[i];
    sum1 += aa[i];
  }
  // Sort the coins in ascending order to easily access the largest ones.
  sort(aa, aa + n);
  /**
   * @block
   * @brief Greedily takes the largest coins until their sum is greater than the rest.
   * @invariant 'sum2' holds the sum of the 'cont' largest coins.
   * @pre 'aa' is sorted, and 'sum1' is the total sum of all coins.
   * @post 'cont' holds the minimum number of coins needed to have a sum strictly greater than the remaining coins.
   */
  for (int i = n - 1; i >= 0; i--) {
    sum2 += aa[i];
    // 'diff' is the sum of the remaining coins.
    diff = sum1 - sum2;
    cont++;
    // Check if the sum of coins taken is strictly greater than the sum of the remaining coins.
    if (sum2 > diff)
      break;
  }
  cout << cont << endl;
}
