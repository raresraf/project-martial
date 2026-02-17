/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A, "Tram".
 * @details This program calculates the minimum required capacity for a tram by
 * tracking the number of passengers over a series of stops. This implementation
 * first reads all passenger data into arrays before simulating the process.
 *
 * @algorithm Iterative simulation with pre-loaded data arrays.
 * @description The program first stores all exit (`out`) and entry (`in`) counts
 * in two separate arrays. It then simulates the passenger flow. The logic in the
 * simulation loop is unconventional but aims to track the current passenger count
 * and the maximum count seen so far.
 * @complexity
 * - Time: O(N), where N is the number of stops, due to two separate loops for
 *   reading data and for simulation.
 * - Space: O(N), to store the `out` and `in` arrays.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  // The variable 'min' is poorly named; it tracks the current passenger count.
  int min = 0; 
  cin >> n;
  int out[n], in[n];

  // Pre-load all stop data into arrays.
  for (int i = 0; i < n; i++) {
    cin >> out[i] >> in[i];
  }

  // 'x' tracks the maximum passenger count found so far. Initialize with the
  // count after the first stop.
  int x = in[0];

  /**
   * @brief This loop simulates the passenger flow across the stops.
   * @note The logic is convoluted. `min` carries a running total of passengers,
   * but the updates are staggered and difficult to follow intuitively.
   */
  for (int i = 0; i < n - 1; i++) {
    // `min` at the start of iteration `i` holds the passenger count after
    // leaving stop `i`.
    min += in[i];

    // Update `x` (max capacity) if the current count is higher.
    if (min >= x)
      x = min;
    
    // Subtract passengers exiting at the *next* stop (`i+1`) to prepare for
    // the next iteration's state.
    min = min - out[i + 1];
  }

  // Output the maximum passenger count found.
  cout << x;
  return 0;
}