/**
 * @file 116A.CPP
 * @brief An implementation for the "Tram" problem.
 * @details This code attempts to find the minimum tram capacity required by simulating
 * passenger flow. It does so by pre-loading all data, calculating passenger counts at each
 * stop, storing them, and then finding the maximum value.
 * @note The implementation is non-standard and more complex than necessary. A simpler O(N)
 * time, O(1) space solution exists by keeping a running count and a running max.
 *
 * @algorithm The algorithm first reads all exit and entry data for `n` stops into arrays.
 * It then iterates from the second to the second-to-last stop, attempting to calculate
 * the number of passengers after each stop. These values are stored in a vector.
 * Finally, the vector is sorted to find the maximum value, which is presented as the answer.
 * The logic for calculating the passenger count at each step is unconventional as it
 * modifies the input array `b` to store the running total.
 *
 * @time_complexity O(N log N), dominated by the sorting of the `total` vector.
 * @space_complexity O(N), due to the use of arrays `a`, `b`, and the `total` vector to
 * store all stop data and intermediate calculations.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  // Pre-allocates arrays to hold all input data before processing.
  int a[n], b[n], left;
  vector<int> total;
  // Block Logic: Read all `n` pairs of exit/entry data into arrays.
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }

  // This condition assumes the first stop has no exits and the last has no entries.
  if (a[0] == 0 && b[n - 1] == 0) {
    // The first passenger count is simply the number who entered at the first stop.
    total.push_back(b[0]);

    /**
     * Block Logic: This loop calculates the passenger count at each subsequent stop.
     * It uses a confusing state-passing mechanism by modifying the `b` array in place.
     */
    for (int i = 1; i < (n - 1); i++) {
      // 'left' is an attempt to calculate remaining passengers. It uses the
      // previously modified value of b[i-1] (now a running total) as the starting point.
      left = b[i - 1] - a[i];
      int x = left + b[i];
      b[i] = x; // Overwrite the original input `b[i]` with the new total.
      total.push_back(x);
    }

    // Sort the collected passenger counts to find the maximum.
    sort(total.begin(), total.end());
    // int m= total.end()-1;
    // Block Logic: Extract and print the maximum value from the sorted vector.
    vector<int>::iterator itr = total.end();
    cout << *(itr - 1) << endl;
  }
  return 0;
}
