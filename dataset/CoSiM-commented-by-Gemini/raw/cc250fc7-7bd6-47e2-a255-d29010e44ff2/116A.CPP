/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem.
 * @details This program calculates the minimum required capacity of a tram by simulating
 * passenger flow over a series of stops. It tracks the number of passengers inside the
 * tram at each stop and finds the maximum number of passengers present at any one time.
 *
 * Algorithm:
 * 1. Read the total number of stops, 'n'.
 * 2. For each stop, read the number of passengers exiting ('a') and entering ('b').
 * 3. Calculate the number of passengers inside the tram after each stop. This is
 *    done using a recurrence relation: passengers_after_stop_i = passengers_after_stop_(i-1) - a_i + b_i.
 *    This is equivalent to calculating the prefix sums of the net passenger change at each stop.
 * 4. The minimum required capacity is the maximum value in the array of passenger counts after each stop.
 *
 * Time Complexity: O(N), where N is the number of stops, due to the loops for input and calculation.
 * Space Complexity: O(N) to store the passenger counts for each stop.
 *
 * Note: The code uses Variable Length Arrays (VLAs), a non-standard C++ extension.
 */

#include <iostream>

using namespace std;

int main() {
  int n; // Number of tram stops
  int c = 0; // Variable to store the minimum required capacity (maximum simultaneous passengers)

  // Read the total number of stops.
  cin >> n;

  // Use VLAs to store passenger data. a[i] = exiting, b[i] = entering.
  // d[i] will store the number of passengers in the tram after stop i.
  int a[n], b[n], d[n];

  /**
   * Block Logic: Read all passenger exit and entry data for each stop.
   * Pre-condition: 'n' is the number of stops.
   * Post-condition: a[] and b[] arrays are populated.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }

  // Calculate passenger count after the first stop. The tram is initially empty.
  // Per the problem, a[0] (exiting at first stop) is always 0.
  d[0] = b[0] - a[0];

  /**
   * Block Logic: Calculate the number of passengers after each subsequent stop.
   * This is a prefix sum calculation on the net change of passengers at each stop.
   * Invariant: d[i] holds the total passengers after stop 'i'.
   */
  for (int i = 1; i < n; i++) {
    d[i] = d[i - 1] - a[i] + b[i];
  }

  /**
   * Block Logic: Find the maximum number of passengers ever on the tram.
   * This maximum value determines the required capacity.
   * Invariant: 'c' holds the maximum value found in d[0...i-1].
   */
  for (int i = 0; i < n; i++) {
    if (d[i] >= c) {
      c = d[i];
    }
  }

  // Output the calculated minimum capacity.
  cout << c << endl;

  return 0;
}
