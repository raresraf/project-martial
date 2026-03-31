/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces (Problem 116A).
 * This program calculates the minimum capacity a tram must have to accommodate
 * all passengers at any given time, given the number of passengers exiting and
 * entering at each stop.
 *
 * Algorithm: Simulation. The program simulates the number of passengers inside
 * the tram at each stop and keeps track of the maximum value encountered.
 * Time Complexity: O(N), where N is the number of tram stops, as each stop
 *                  is processed once.
 * Space Complexity: O(N) for storing passenger data in arrays.
 *                  (Note: Uses Variable Length Arrays (VLAs), a C99 feature,
 *                  which is a non-standard extension in C++).
 */

#include <iostream> // @brief Required for standard input/output streams (cin, cout).

using namespace std;

int main() {
  /**
   * @brief Main function to calculate the maximum number of passengers in a tram.
   * Reads the number of stops, then for each stop, reads passengers exiting and entering.
   * It simulates the passenger count at each stop and determines the peak occupancy.
   */

  int n; // @brief Number of tram stops.
  int ans = 0; // @brief Stores the maximum number of passengers found, initialized to 0.

  cin >> n; // @brief Read the number of tram stops.

  // Block Logic: Declare Variable Length Arrays (VLAs) to store passenger data.
  // 'a' stores passengers exiting, 'b' stores passengers entering, 'tmp' stores current passenger count.
  // Note: VLAs are a C99 feature and non-standard in C++.
  int a[n] = {0}; // @brief Array for passengers exiting at each stop.
  int b[n] = {0}; // @brief Array for passengers entering at each stop.
  int tmp[n] = {0}; // @brief Array to store passenger count after each stop.

  // Block Logic: Read passenger data for each stop.
  // Loop uses 1-based indexing (i from 1 to n) for arrays 'a' and 'b'.
  for (int i = 1; i <= n; ++i) {
    cin >> a[i] >> b[i]; // Inline: Read passengers exiting and entering for stop 'i'.
  }

  // Block Logic: Initialize passenger count after the first stop.
  // 'tmp[0]' stores the passenger count after the first stop (only those entering).
  tmp[0] = b[1]; // @brief Passengers after the first stop.

  // Block Logic: Calculate passenger count at subsequent stops.
  // Loop uses 1-based indexing for 'i', effectively processing stops from 2 to n.
  // Pre-condition: 'tmp[i-1]' holds the passenger count before processing stop 'i'.
  // Invariant: 'tmp[i-1]' (after calculation) holds passengers after stop 'i', and 'b[i]' is updated.
  for (int i = 2; i <= n; ++i) {
    // Block Logic: Calculate current passengers: (previous passengers - exiting) + entering.
    // Note: 'b[i-1]' here seems to represent the passengers *inside* the tram just before stop 'i'.
    // This value is updated in 'tmp[i-1]' to reflect passengers *after* stop 'i'.
    tmp[i - 1] = b[i - 1] - a[i]; // Inline: Passengers remaining after exits.
    tmp[i - 1] += b[i]; // Inline: Add new passengers entering.

    // Note: The original code reassigns `b[i] = tmp[i-1]`, which might be a way to
    // carry forward the current passenger count for the next iteration's calculation.
    b[i] = tmp[i - 1]; // @brief Update 'b[i]' with the current passenger count.
  }

  // Block Logic: Find the maximum passenger count among all stops.
  // Loop uses 0-based indexing for 'tmp' array.
  for (int i = 0; i < n; ++i) {
    if (tmp[i] > ans) { // Inline: If current stop's passenger count is greater than current max.
      ans = tmp[i]; // Inline: Update maximum.
    }
    // cout<<tmp[i]; // @brief Commented out debug print statement.
  }

  cout << ans; // @brief Print the maximum number of passengers.

  return 0; // @brief Indicates successful program termination.
}
