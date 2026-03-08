/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity for a tram.
 *
 * @details
 * This program solves the "Tram" problem by simulating passenger flow across
 * a number of stops. It uses a two-pass approach. The first pass simulates
 * the passenger count stop-by-stop, storing the number of passengers after
 * each stop in an array. The second pass then iterates through this array to
 * find the maximum passenger count, which corresponds to the required tram capacity.
 *
 * @section
 * Algorithm: Two-pass simulation and search.
 * Time Complexity: O(N), where N is the number of stations, due to two separate loops.
 * Space Complexity: O(N), to store the passenger count at each stop in an intermediate array.
 * @note A more optimal O(1) space solution could achieve this in a single pass
 * without the intermediate array.
 */
#include <iostream>
using namespace std;
int main() {
  // ar: An array to store the passenger count after each stop. Fixed size assumes n <= 1000.
  int ar[1000];
  // enter, exit: passengers entering/exiting at a stop.
  // r: running total of current passengers on the tram.
  int enter, exit, r = 0;
  // station: the total number of stops.
  int station;
  cin >> station;

  /**
   * @brief First pass: Simulate and record passenger counts.
   *
   * Logic: This loop reads the passenger flow for each station, updates the
   * current passenger count 'r', and stores this count in the 'ar' array.
   */
  for (int i = 0; i < station; i++) {
    cin >> exit >> enter;
    // Update the running total of passengers (exiting first, then entering).
    r = (r - exit) + enter;
    // Record the passenger count after this stop.
    ar[i] = r;
  }

  // 'temp' will hold the maximum passenger count found, i.e., the required capacity.
  int temp = 0;
  /**
   * @brief Second pass: Find the maximum passenger count.
   *
   * Logic: This loop iterates through the recorded passenger counts for each stop
   * and finds the peak value.
   */
  for (int i = 0; i < station; i++) {

    // Pre-condition: Check if the current stop's passenger count is the highest so far.
    if (ar[i] > temp) {
      temp = ar[i];
    } else
      // This 'continue' is redundant but harmless.
      continue;
  }
  // Output the maximum capacity required.
  cout << temp;
}
