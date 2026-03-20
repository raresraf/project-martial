/**
 * @file 116A.CPP
 * @brief Solution for Codeforces problem 116A, "Tram".
 * @details This program calculates the minimum required capacity of a tram.
 * This particular implementation simulates the passenger count at each stop,
 * stores these counts in an array, and then sorts the array to find the maximum.
 *
 * @section Algorithm
 * The approach is to find the peak passenger occupancy by recording the state
 * after each stop.
 *
 * 1. Read the number of stops, `t`.
 * 2. Read all passenger exit and entry data into a 2D array `a`.
 * 3. Initialize a counter `k` for the current number of passengers to 0.
 * 4. In a loop, simulate the process for each stop:
 *    a. Update the current passenger count `k` by subtracting exiting passengers
 *       and adding entering ones.
 *    b. Store the updated count `k` in a separate array, `a1`.
 * 5. After simulating all stops, sort the array `a1` which contains the
 *    passenger counts at each stage.
 * 6. The maximum value in `a1` (which is the last element after sorting) is the
 *    minimum required capacity. This value is printed.
 *
 * This method is less optimal than finding the maximum on-the-fly, as it
 * incurs the overhead of a sort operation.
 *
 * @subsection Complexity
 * - Time Complexity: O(t log t), where `t` is the number of stops. This is
 *   dominated by the `std::sort` function.
 * - Space Complexity: O(t), due to the two arrays `a` and `a1` used to store
 *   the input data and intermediate passenger counts.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  // t: number of stops. i, j: loop counters. k: current passenger count.
  int t, i, j, k;
  // a: stores exit/entry data for each stop.
  // a1: stores the total number of passengers after each stop.
  int a[1001][2], a1[1001];
  cin >> t;
  // This loop reads all the stop data first.
  for (i = 0; i < t; i++) {
    for (j = 0; j < 2; j++) {
      cin >> a[i][j];
    }
  }

  k = 0; // Initialize current passenger count.
  /**
   * @brief This loop simulates the tram's state at each stop.
   * It calculates the number of passengers in the tram after each stop
   * and stores it in the `a1` array.
   */
  for (i = 0; i < t; i++) {
    k -= a[i][0]; // Passengers exit.
    k += a[i][1]; // Passengers enter.
    a1[i] = k;    // Record the current passenger count.
  }

  // Sort the array of passenger counts in ascending order.
  sort(a1, a1 + t);
  // The last element of the sorted array is the maximum passenger count.
  cout << a1[t - 1] << endl;
  return 0;
}
