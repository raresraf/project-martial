/**
 * @file 116A.CPP
 * @brief An incorrect attempt to solve the "Tram" problem (Codeforces 116A).
 *
 * @algorithm (Intended)
 * The problem is to find the maximum capacity required for a tram operating over
 * `n` stops. At each stop, a number of people exit and a number of people enter.
 * The required capacity is the maximum number of people on the tram at any one time.
 *
 * A correct algorithm would be:
 * 1. Initialize `current_passengers = 0` and `max_passengers = 0`.
 * 2. Loop `n` times, once for each stop.
 * 3. In each loop, read the exiting (`a`) and entering (`b`) passenger counts.
 * 4. Update the current passenger count: `current_passengers = current_passengers - a + b`.
 * 5. Update the maximum capacity: `max_passengers = max(max_passengers, current_passengers)`.
 * 6. The final `max_passengers` is the answer.
 *
 * @warning
 * FLAWED IMPLEMENTATION: The logic in this code is incorrect and does not solve
 * the problem. It reads all inputs into an array and then performs a series of
 * arithmetic operations that overwrite the array's contents. This process does
 * not correctly model the changing number of passengers on the tram and produces
 * a wrong answer.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int m = INT_MIN; // Intended to store the max capacity.
  // Using a non-standard Variable Length Array (VLA) to store all inputs.
  int num[n * 2];
  // First loop: Read all exit/entry numbers into a single flat array.
  for (int i = 0; i < n * 2; i++) {
    cin >> num[i];
  }
  /**
   * @brief Second loop: This section contains the flawed processing logic.
   * It attempts to calculate the passenger count but overwrites its own input
   * array with nonsensical calculations. This does not correctly track the
   * number of people on the tram.
   */
  for (int i = 0; i < n * 2; i++) {
    int k = 0; // Unused variable.
    if (i % 2 == 0) { // Corresponds to an exit count `a_i`.
      // Incorrectly overwrites the entry count `b_i` with `a_i + b_i`.
      num[i + 1] = num[i] + num[i + 1];
    } else { // Corresponds to an entry count `b_i`.
      // Incorrectly overwrites the next stop's exit count with a meaningless value.
      num[i + 1] = num[i] - num[i + 1];
    }
    // Compares the incorrect calculation with the max value.
    if (num[i + 1] > m)
      m = num[i + 1];
  }
  // The output `m` is not the correct answer.
  cout << m << endl;

  return 0;
}
