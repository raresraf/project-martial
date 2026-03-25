#include <iostream>
#include <vector> // To replace VLA with a standard container.

/**
 * @file 116A.CPP
 * @brief Solution for a simulation problem, "Tram".
 *
 * Problem Statement:
 * A tram makes a certain number of stops. At each stop, some passengers exit
 * and some enter. The task is to calculate the minimum possible capacity of
 * the tram, which is equivalent to finding the maximum number of passengers
 * inside the tram at any single point in time. The tram is empty before the
 * first stop.
 *
 * Algorithm:
 * The program simulates the tram's journey stop by stop.
 * 1. It maintains a running count of the current number of passengers on the
 *    tram and a variable to track the maximum number of passengers seen so far.
 * 2. It iterates through each stop. At each stop, it updates the current
 *    passenger count by subtracting the number of exiting passengers and
 *    adding the number of entering passengers.
 * 3. After each update, it compares the current passenger count with the
 *    maximum recorded so far and updates the maximum if the current count
 *    is higher.
 * 4. The final maximum value is the minimum required capacity.
 *
 * @note The original code uses Variable Length Arrays (VLAs) like `int in[number]`,
 * which are a non-standard C++ extension. A `std::vector` is the standard C++
 * way to handle dynamic arrays. An even more memory-efficient approach would
 * be to process the input for each stop immediately in a single loop,
 * avoiding the need to store all inputs in arrays.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(N) due to storing all inputs in arrays.
 */

using namespace std;

int main() {
  int number;
  cin >> number;

  // Using VLAs, which is a non-standard C++ extension.
  // std::vector<int> in(number), out(number); would be the standard way.
  int in[number], out[number], i;

  // total: current number of passengers on the tram.
  // minCap: the maximum number of passengers observed so far.
  int total = 0, minCap = 0;

  // First loop: Read and store all the passenger data for each stop.
  for (i = 0; i < number; i++) {
    cin >> out[i] >> in[i];
  }

  // Second loop: Simulate the journey and calculate capacity.
  // Invariant: 'total' is the number of passengers after stop 'i-1'.
  // 'minCap' is the max passengers seen up to stop 'i-1'.
  for (i = 0; i < number; i++) {
    // Update the current number of passengers.
    total = total + in[i] - out[i];

    // Pre-condition: Check if the new passenger count exceeds the max seen so far.
    if (total > minCap) {
      minCap = total;
    }
  }

  cout << minCap;
  return 0;
}
