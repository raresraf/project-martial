/**
 * @file 116A.CPP
 * @brief Simulates a tram journey to find the maximum number of passengers onboard at any given time.
 *
 * Algorithm: Iterative tracking of current passenger count and maximum observed count.
 * Time Complexity: O(N) where N is the number of tram stops, due to a single pass through the stops.
 * Space Complexity: O(1) as only a few variables are used to store state.
 */
#include <bits/stdc++.h>
#define ll long long int // Functional Utility: Defines 'll' as a shorthand for 'long long int' for competitive programming.
#define pb push_back // Functional Utility: Defines 'pb' as a shorthand for 'push_back', commonly used with vectors.

using namespace std;

/**
 * @brief Main function to simulate the tram journey and determine maximum occupancy.
 * @pre Input consists of the number of stops (n), followed by 'n' pairs of integers (passengers exiting, passengers entering).
 * @post Prints the maximum number of passengers observed in the tram during the entire journey.
 */
int main() {
  int n, a, b, t = 0, h = -1; // 'n': number of stops, 'a': passengers exiting, 'b': passengers entering, 't': current passengers, 'h': max passengers.

  // Functional Utility: Reads the total number of tram stops.
  cin >> n;

  /**
   * Block Logic: Iterates through each tram stop, updates the current passenger count, and tracks the maximum.
   * Invariant: At the start of each iteration, 't' represents the number of passengers after the previous stop.
   *            At the end of each iteration, 't' reflects the passenger count after the current stop's changes.
   *            'h' always stores the highest passenger count encountered so far.
   */
  for (int i = 0; i < n; i++) {
    // Functional Utility: Reads the number of passengers exiting and entering at the current stop.
    cin >> a >> b;

    // Functional Utility: Updates the current passenger count.
    t = t - a; // Passengers exit.
    t = t + b; // Passengers enter.

    /**
     * Block Logic: Updates the maximum passenger count if the current count exceeds it.
     * Pre-condition: 't' holds the current passenger count, 'h' holds the maximum so far.
     * Invariant: 'h' will be updated to 't' if 't' is greater than 'h'.
     */
    if (t > h)
      h = t;
  }

  // Functional Utility: Prints the overall maximum number of passengers found.
  cout << h << endl;
  // Indicates successful program execution.
  return 0;
}
