/**
 * @file 1343C.CPP
 * @brief Solves the "Alternating Subsequence" problem with a greedy approach.
 *
 * The goal is to find a subsequence with the maximum possible sum, where the signs
 * of its elements must alternate.
 *
 * @b Algorithm: Greedy Strategy
 *   The core idea is to partition the array into contiguous blocks of numbers with
 *   the same sign. To maximize the sum, we should pick exactly one element from each
 *   of these blocks: the largest one. This code iterates through the array, identifying
 *   these blocks of alternating signs and adding the maximum element from each to the total sum.
 *
 * @b Time_Complexity: O(N) per test case, as the array is traversed once.
 * @b Space_Complexity: O(N) per test case to store the input array.
 */
#include <bits/stdc++.h>
using namespace std;
#define endl "\n"

int main() {
  // Fast I/O for competitive programming.
  ios_base ::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int t, n;
  cin >> t;
  while (t--) {
    cin >> n;
    int a[n];
    for (int i = 0; i < n; i++) {
      cin >> a[i];
    }

    long long sum = 0;
    int i = 0;

    /**
     * @brief Main loop to traverse the array and process blocks of same-signed numbers.
     * This loop correctly partitions the array into alternating blocks of positive and
     * negative numbers and finds the max of each block to add to the sum.
     */
    while (i < n) {
      int maxm = INT_MIN;

      // Find the largest element in the current contiguous block of negative numbers.
      while (i < n && a[i] < 0) {
        maxm = max(maxm, a[i]);
        i++;
      }
      if (maxm != INT_MIN) {
        sum += maxm;
      }

      maxm = INT_MIN;

      // Find the largest element in the current contiguous block of positive numbers.
      while (i < n && a[i] > 0) {
        maxm = max(maxm, a[i]);
        i++;
      }
      if (maxm != INT_MIN) {
        sum += maxm;
      }
    }
    cout << sum << endl;
  }

  return 0;
}