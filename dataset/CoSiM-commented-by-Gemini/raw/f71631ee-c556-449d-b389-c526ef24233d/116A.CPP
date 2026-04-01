/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces (116A).
 * @details The program calculates the minimum possible capacity of a tram based on the number of passengers
 * entering and exiting at each stop. It tracks the number of passengers inside the tram at any
 * given time and finds the maximum value this count reaches.
 *
 * Algorithm:
 * 1. Read the number of tram stops `n`.
 * 2. Initialize `current_passengers` to 0 and `min_capacity` to 0.
 * 3. Loop `n` times for each stop:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the number of passengers inside the tram: `current_passengers = current_passengers - a + b`.
 *    c. Update the minimum required capacity by taking the maximum of `min_capacity` and `current_passengers`.
 * 4. After the loop, `min_capacity` will hold the maximum number of passengers that were ever on the tram at one time, which is the minimum capacity required.
 * 5. Print the `min_capacity`.
 *
 * Time Complexity: O(N), where N is the number of stops, as we iterate through the stops once.
 * Space Complexity: O(1), as we only use a few variables to store the state.
 */
#include <algorithm>
#include <iostream>
using namespace std;

/**
 * @brief Main logic to solve the Tram problem.
 * Reads passenger data for each stop, tracks the current number of passengers,
 * and determines the maximum number of passengers ever on the tram.
 */
void solved() {
  int a, b; // a: passengers exiting, b: passengers entering
  int n; // number of stops
  cin >> n;
  int temp = 0; // Represents the current number of passengers on the tram.
  int MAX = 0;  // Represents the minimum capacity required, initialized to 0.

  /**
   * Block Logic: This loop simulates the tram's journey stop by stop.
   * Invariant: At the beginning of each iteration, `temp` holds the number of passengers
   * on the tram as it arrives at the current stop. `MAX` holds the maximum number of
   * passengers observed so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Update the number of passengers after accounting for exits and entries.
    temp = temp + b - a;
    // Update the required capacity if the current number of passengers is the new maximum.
    MAX = max(temp, MAX);
  }
  // Output the minimum capacity needed for the tram.
  cout << MAX;
}

int main() {
  solved();
  return 0;
}
