/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" capacity problem from a programming contest.
 * @details The program calculates the minimum required capacity for a tram by
 * simulating the passenger flow over 'n' stops. This implementation uses a
 * two-pass approach: first, it reads and stores all passenger data, and second,
 * it processes the stored data to find the peak passenger count.
 *
 * Algorithm:
 * 1. Read the total number of stops, 'n'.
 * 2. Create two arrays to store the number of passengers entering and exiting
 *    at each stop.
 * 3. A first loop reads all 'n' pairs of exit/entry counts and populates the arrays.
 * 4. Initialize `current_passengers` and `max_capacity` to 0.
 * 5. A second loop iterates through the stored data from stop 1 to 'n'.
 *    a. Update `current_passengers` using the stored entry and exit counts for that stop.
 *    b. If `current_passengers` exceeds `max_capacity`, update `max_capacity`.
 * 6. After the second loop, `max_capacity` holds the minimum required capacity.
 *
 * Time Complexity: O(n), due to the two separate passes over the 'n' stops.
 * Space Complexity: O(n) to store the entry and exit counts in arrays.
 */
#include <iostream>

using namespace std;

int main() {
  int number;
  cin >> number;
  // Using Variable Length Arrays (VLAs), a non-standard C++ extension.
  int in[number], out[number], i, total = 0, minCap = 0;

  /**
   * @block
   * @description First pass: Read and store all passenger entry and exit data
   * for each of the 'number' stops.
   */
  for (i = 0; i < number; i++) {
    cin >> out[i] >> in[i];
  }

  /**
   * @block
   * @description Second pass: Process the stored data to simulate the passenger
   * flow and find the maximum concurrent passenger count.
   */
  for (i = 0; i < number; i++) {
    // Update the current number of passengers on the tram.
    total = total + in[i] - out[i];
    // If the current count is a new peak, update the required capacity.
    if (total > minCap)
      minCap = total;
  }

  // Output the final calculated minimum capacity.
  cout << minCap;
  return 0;
}