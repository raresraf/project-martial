/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from competitive programming.
 *
 * @details
 * This program calculates the minimum possible capacity of a tram. It is given
 * the number of stops and for each stop, the number of passengers exiting and
 * entering. The minimum capacity must be equal to the maximum number of
 * passengers inside the tram at any point in time.
 *
 * Algorithm:
 * 1. Read the total number of stops, n.
 * 2. Initialize `currentCap` (passengers currently inside) and `maxCap`
 *    (the peak number of passengers seen so far) to 0.
 * 3. Loop n times to simulate the tram's journey from stop to stop.
 *    a. For each stop, read the number of exiting passengers (`a`) and
 *       entering passengers (`b`).
 *    b. Update the `currentCap` by adding the entering passengers and
 *       subtracting the exiting ones.
 *    c. Compare the updated `currentCap` with `maxCap`. If `currentCap` is
 *       larger, update `maxCap` to this new peak value.
 * 4. After iterating through all stops, `maxCap` will hold the minimum
 *    capacity required, which is then printed.
 *
 * Time Complexity: O(N), where N is the number of stops, because the program iterates through each stop once.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  int maxCap = 0;     // Tracks the maximum number of passengers seen so far.
  int currentCap = 0; // Tracks the current number of passengers on the tram.
  cin >> n;           // Read the number of stops.

  /**
   * @brief Loop to simulate the tram's journey through all the stops.
   * Pre-condition: 'n' is the number of stops.
   * Invariant: `currentCap` holds the number of passengers after stop `i-1`,
   *            and `maxCap` holds the peak passenger count up to that point.
   */
  for (int i = 0; i < n; i++) {
    int a, b; // 'a' is exiting, 'b' is entering.
    cin >> a >> b;

    // Update the current number of passengers.
    // Although the order is `+b` then `-a`, it's equivalent to `-a` then `+b`.
    // The number of people on the tram becomes (current - exiting + entering).
    currentCap += b;
    currentCap -= a;

    // Check if the current passenger count exceeds the maximum capacity recorded
    // so far.
    if (currentCap >= maxCap) {
      maxCap = currentCap;
    }
  }

  // Print the minimum required capacity.
  cout << maxCap << endl;
  return 0;
}