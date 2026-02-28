/**
 * @file 116A.CPP
 * @brief An incorrect and logically flawed attempt to solve the "Tram" problem from Codeforces.
 * @details The goal of the problem is to find the maximum number of passengers on a tram
 * at any single point in time to determine its minimum required capacity. This code
 * attempts to do so but contains several logical errors in its simulation.
 *
 * @algorithm Flawed Simulation
 * 1. The code first reads all entry and exit data for all stops into two arrays,
 *    which is an unnecessary use of memory.
 * 2. It attempts to calculate the passenger count with an initial formula that is
 *    logically incorrect.
 * 3. It then iterates, but the logic for updating the passenger count and tracking
 *    the maximum is flawed.
 * 4. A final loop incorrectly finds the maximum of all individual entry/exit values,
 *    which does not correspond to the problem's requirement of finding the peak
 *    *cumulative* passenger count.
 *
 * @note This solution does not produce the correct answer for the problem.
 */

#include <iostream>

int main() {
  int n;
  std::cin >> n;
  
  // The arrays `a` and `b` store all exit and entry counts upfront.
  // A correct solution would process these stop-by-stop.
  // Note: `a[n]` and `b[n]` are Variable Length Arrays, a non-standard C++ extension.
  int a[n], b[n];
  
  // `max` is intended to store the maximum capacity.
  // `j` is intended to track the current number of passengers.
  int max = 0, j;
  
  /**
   * @brief This loop only reads all the data points into arrays before any calculation.
   */
  for (int i = 0; i < n; i++) {
    std::cin >> a[i] >> b[i];
  }

  /**
   * @brief Incorrect initial calculation.
   * This line attempts to calculate the passenger count after the second stop but uses
   * incorrect indices and logic. A correct approach would be:
   * current_passengers = (0 - a[0] + b[0]) - a[1] + b[1];
   */
  j = b[0] - a[1] + b[1];

  /**
   * @brief Flawed simulation loop.
   * This loop starts from the 3rd stop and continues the incorrect calculation.
   * It also checks for the max value *before* updating the passenger count for the
   * current stop, which is off-by-one logic.
   */
  for (int i = 2; i < n; i++) {
    if (j > max) {
      max = j;
    }
    j = j - a[i] + b[i];
  }

  /**
   * @brief Incorrect final check.
   * This loop does not simulate the passenger flow. Instead, it incorrectly finds the
   * single largest number among all individual exit `a[i]` and entry `b[i]` values.
   * This is unrelated to the cumulative number of passengers on the tram over time.
   */
  for (int i = 0; i < n; i++) {
    if (a[i] > max) {
      max = a[i];
    }

    if (b[i] > max) {
      max = b[i];
    }
  }

  // This prints an incorrect result due to the multiple logical flaws above.
  std::cout << max;

  return 0;
}
