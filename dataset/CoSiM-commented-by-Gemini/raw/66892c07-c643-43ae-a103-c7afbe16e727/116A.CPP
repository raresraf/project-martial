#include <iostream>
using namespace std;

/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem from Codeforces (Problem 116A).
 * @details This program calculates the minimum required capacity of a tram by
 * simulating the change in passenger numbers at each stop.
 *
 * Algorithm:
 * The program reads the number of stops, `n`. It then iterates through each stop,
 * reading the number of passengers who exit and enter. A running total of the
 * current number of passengers inside the tram is maintained. After each stop,
 * this current passenger count is compared with the maximum passenger count seen
 * so far. If the current count is higher, the maximum is updated. This maximum
 * value, determined after all stops have been processed, represents the minimum
 * capacity the tram must have.
 *
 * Time Complexity: O(n), where `n` is the number of stops, because there is a single
 * pass through all the stops.
 * Space Complexity: O(1), as the program uses a fixed amount of memory regardless of
 * the number of stops.
 */
int main() {
  int n;
  // Read the number of tram stops.
  cin >> n;
  // `current_passengers` tracks the number of people currently on the tram.
  // `max_capacity` stores the maximum number of passengers observed at any one time.
  int current_passengers = 0, exit_count, enter_count, max_capacity = 0;

  /**
   * @brief Loop through each stop to update passenger counts.
   * Pre-condition: `n` is the number of stops remaining to be processed.
   * Invariant: `current_passengers` holds the number of people on the tram
   * just before arriving at the current stop. `max_capacity` holds the peak
   * passenger count observed up to the previous stop.
   */
  while (n--) {
    // Read the number of passengers exiting and entering at the current stop.
    cin >> exit_count >> enter_count;

    // Update the number of passengers on the tram.
    current_passengers -= exit_count;
    current_passengers += enter_count;

    // If the current number of passengers exceeds the max capacity recorded so far, update the max.
    if (current_passengers > max_capacity)
      max_capacity = current_passengers;
  }
  // Output the minimum required capacity.
  cout << max_capacity;
}