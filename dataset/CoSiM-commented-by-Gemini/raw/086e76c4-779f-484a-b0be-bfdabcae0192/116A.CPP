/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 * @details This program solves the "Tram" problem from Codeforces by simulating
 * the passenger count at each stop to find the peak number of passengers
 * on board at any single time.
 *
 * @see https://codeforces.com/problemset/problem/116/A
 *
 * Algorithm:
 * 1. Read the total number of stops.
 * 2. Initialize a variable to track the current number of passengers (`rev`) and
 *    another to track the maximum passengers seen so far (`tem`).
 * 3. Loop for each stop:
 *    a. Read the number of passengers exiting and entering.
 *    b. Update the current passenger count by adding the entrants and subtracting the exitees.
 *    c. Compare the current count with the maximum seen so far and update the maximum if needed.
 * 4. After the loop, the maximum value found is the minimum required tram capacity. Print this value.
 *
 * Time Complexity: O(N), where N is the number of stops, due to the single pass through the stops.
 * Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <iostream>
using namespace std;

int main() {
  // S: The total number of stops.
  long int S;
  // exit, enter: Passengers exiting and entering at the current stop.
  long int exit, enter;
  // rev: Tracks the current number of passengers on the tram.
  long int rev = 0;
  // tem: Tracks the maximum number of passengers seen so far, which is the required capacity.
  long int tem = 0;

  cin >> S;

  /**
   * @brief Main loop to simulate the tram's journey.
   *
   * Pre-condition: S holds the number of stops to simulate.
   * Invariant: After each iteration, 'rev' is the current passenger count, and 'tem'
   *            is the maximum count observed up to that point.
   */
  while (S--) {
    cin >> exit >> enter;
    // Update the current number of passengers.
    rev = rev + enter - exit;
    // If the current passenger count is greater than the max recorded, update the max.
    tem = tem >= rev ? tem : rev;
  }

  // Output the minimum required capacity.
  cout << tem;
  return 0;
}
