// CodeForces
// Problem 116A [Tram]

/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A, "Tram".
 * @details This program calculates the minimum possible capacity of a tram by
 * tracking the number of passengers at each stop.
 *
 * Algorithm:
 * The code uses a single-pass simulation approach.
 * 1. It initializes `current_capacity` to 0 to track the number of people
 *    currently on the tram and `max_capacity` to 0 to store the peak
 *    number of passengers observed.
 * 2. It loops through each stop.
 * 3. In each iteration, it reads the number of passengers exiting and entering.
 * 4. It updates `current_capacity` by subtracting the exits and adding the entries.
 * 5. It then compares the updated `current_capacity` with `max_capacity`. If the
 *    current count is higher, `max_capacity` is updated to this new peak.
 * 6. After processing all stops, `max_capacity` holds the minimum capacity the
 *    tram must have, which is then printed.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  long long total_stops, current_capacity = 0, max_capacity = 0, entries = 0,
                         exits = 0;
  cin >> total_stops;

  /**
   * @brief This loop simulates the tram's journey stop by stop.
   */
  while (total_stops--) {
    cin >> exits >> entries;

    // Update the number of passengers currently on the tram.
    current_capacity -= exits;
    current_capacity += entries;

    /**
     * @brief Update the maximum capacity required if the current number of
     * passengers is the highest seen so far.
     */
    if (current_capacity > max_capacity) {
      max_capacity = current_capacity;
    }
  }
  cout << max_capacity;
  return 0;
}