/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 * @details The program calculates the minimum required capacity for a tram based on the number of
 * passengers entering and exiting at each stop.
 *
 * @author (unknown)
 *
 * @algorithm
 * 1. Read the total number of tram stops, `paradas`.
 * 2. Initialize `p` to 0, representing the current number of passengers on the tram.
 * 3. Initialize `minimum` (acting as `max_capacity_needed`) to 0. This variable will track the
 *    maximum number of passengers on the tram at any given time.
 * 4. Loop through each stop:
 *    a. Read the number of passengers exiting (`bajan`) and entering (`suben`).
 *    b. Update the current passenger count: `p = p - bajan + suben`.
 *    c. After passengers enter, check if the new count `p` is the highest seen so far.
 *       If `p` is greater than `minimum`, update `minimum` to `p`.
 * 5. After iterating through all stops, `minimum` holds the peak passenger count, which
 *    is the minimum capacity the tram must have. Print this value.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stops (`paradas`), due to the single loop.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {

  int paradas, bajan, suben, p = 0, minimum = 0;
  // paradas: number of stops
  cin >> paradas;

  /**
   * Block Logic: This loop simulates the tram's journey through all the stops.
   * Pre-condition: `paradas` contains the total number of stops.
   * Invariant: At the start of each iteration, `p` holds the number of passengers arriving at the stop,
   * and `minimum` holds the maximum passenger count observed in previous stops.
   */
  for (int i = 0; i < paradas; i++) {

    cin >> bajan; // Number of passengers exiting
    p = p - bajan;

    cin >> suben; // Number of passengers entering
    p = p + suben;
    /**
     * Block Logic: Update the required capacity.
     * The `minimum` variable here is used to track the maximum number of people
     * on the tram at any point. If the current number of passengers `p` is greater
     * than the max seen so far, we update our required capacity.
     */
    if (p > minimum) {
      minimum = p;
    }
  }

  // After the loop, 'minimum' holds the peak passenger count, which is the required capacity.
  cout << minimum << endl;

  return 0;
}