/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem on Codeforces.
 * @author Nikhil Nagrale (nikhilnagrale2)
 *
 * @details The problem asks for the minimum number of coins one needs to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * Links:
 * - Problem: https://codeforces.com/problemset/problem/160/A
 * - Contest: https://codeforces.com/contest/160/problem/A
 *
 * Algorithm:
 * 1. Read all 'n' coin values and calculate their total sum.
 * 2. To maximize the sum with the minimum number of coins, a greedy approach is used.
 * 3. Sort the coin values in descending order.
 * 4. Iterate through the sorted coins, adding their value to a running sum 'mx' and
 *    incrementing a coin counter.
 * 5. In each iteration, check if the current sum of taken coins ('mx') is strictly
 *    greater than the sum of the remaining coins ('sum - mx').
 * 6. If it is, the loop breaks, and the number of coins taken is the answer.
 *
 * Time Complexity: O(N log N) dominated by the sorting step.
 * Space Complexity: O(N) to store the coin values in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> a(n);
  int sum = 0;
  // Read coin values and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  
  // Sort coins in non-decreasing order.
  sort(a.begin(), a.end());
  // Reverse to get descending order, for the greedy approach.
  reverse(a.begin(), a.end());
  
  int mx = 0;
  int i = 0;
  /**
   * @brief Greedily take the largest coins until their sum is greater than the rest.
   * Pre-condition: 'a' is sorted in descending order. 'sum' is the total value of all coins.
   * Invariant: 'mx' is the sum of the 'i' largest coins.
   */
  for (i = 0; i < n; i++) {
    mx += a[i];
    // Check if the sum of taken coins is strictly greater than the sum of remaining coins.
    if (mx > sum - mx)
      break;
  }
  // Output the minimum number of coins required. 'i' is 0-indexed, so we add 1.
  cout << i + 1 << endl;
  return 0;
}