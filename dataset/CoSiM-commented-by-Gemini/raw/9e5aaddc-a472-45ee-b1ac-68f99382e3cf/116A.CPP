/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program solves a capacity tracking problem. It's given the number of
 * passengers who exit and enter at each of a tram's `n` stops. The goal is to
 * find the maximum number of passengers on the tram at any one time, which
 * corresponds to the minimum capacity the tram must have.
 *
 * @algorithm The program reads the number of people exiting (`a[i]`) and entering
 * (`b[i]`) at each stop. It attempts to calculate the number of people on the
 * tram after each stop (`c[i]`) and find the maximum among these values.
 *
 * @note The implementation is buggy. The loops for calculating the capacity
 * (`c[i]`) and finding the maximum value both run up to `n-2` (`i < n - 1`),
 * completely ignoring the last stop. This will produce an incorrect result if
 * the maximum capacity is reached at or after the second-to-last stop.
 * A correct implementation would iterate up to `n-1` and update the maximum
 * capacity inside the same loop where the current capacity is calculated.
 *
 * @time_complexity O(N), where N is the number of stops, due to the loops.
 * @space_complexity O(N) due to the use of three arrays (`a`, `b`, `c`) to store
 * the data for all stops. A more space-efficient O(1) solution is possible
 * by processing the data stop-by-stop without storing all of it.
 */
#include <iostream>

int main() {
  int i;         // Loop counter.
  int n;         // The number of tram stops.
  int a[1005];   // Array to store the number of exiting passengers at each stop.
  int b[1005];   // Array to store the number of entering passengers at each stop.
  int c[1005];   // Array to store the number of passengers on the tram after each stop.
  int max = 0;   // Variable to track the maximum capacity needed.
  
  std::cin >> n;
  
  // Read all the exit and entry data for each stop.
  for (i = 0; i < n; i++)
    std::cin >> a[i] >> b[i];
  
  // Capacity after the first stop (assumes it starts empty, a[0] is ignored).
  c[0] = b[0];
  
  /**
   * @brief Loop to calculate the tram's occupancy after each stop.
   * @warning This loop is incorrect. It stops at `n-2` (`i < n-1`), failing to
   * calculate the capacity for the last stop.
   */
  for (i = 1; i < n - 1; i++)
    c[i] = c[i - 1] - a[i] + b[i];
  
  /**
   * @brief Loop to find the maximum occupancy.
   * @warning This loop is also incorrect. It stops at `n-2` (`i < n-1`),
   * failing to check the occupancy calculated for the second-to-last stop and
   * completely ignoring the last stop.
   */
  for (i = 0; i < n - 1; i++) {
    if (c[i] > max)
      max = c[i];
  }
  
  // Print the calculated maximum value.
  std::cout << max << '\n';
  return 0;
}
