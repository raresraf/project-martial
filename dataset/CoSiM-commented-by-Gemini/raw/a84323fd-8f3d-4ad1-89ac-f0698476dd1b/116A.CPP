/**
 * @file 116A.CPP
 * @brief An attempted solution for CodeForces Problem 116A - "Tram".
 *
 * @warning The logic in this file is incorrect and does not solve the problem.
 *          The final output is `n` (the number of stops), not the required
 *          tram capacity. The internal calculations are also flawed.
 *
 * @details
 * The "Tram" problem requires calculating the maximum number of passengers on a tram
 * at any given time to determine its minimum required capacity. The correct algorithm is to
 * simulate the passenger count stop-by-stop.
 *
 * Correct Algorithm:
 * 1. Initialize `current_passengers = 0` and `max_passengers = 0`.
 * 2. Loop `n` times for each stop:
 *    a. Read the number of exiting (`exit`) and entering (`enter`) passengers.
 *    b. Update `current_passengers = current_passengers - exit + enter`.
 *    c. Update `max_passengers = max(max_passengers, current_passengers)`.
 * 3. The result is `max_passengers`.
 *
 * This implementation attempts a different, incorrect approach.
 *
 * @time_complexity O(N) due to the loops.
 * @space_complexity O(1) as the array size is fixed, though a VLA would be O(N).
 */
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int n;
  int m = 0; // Intended to be the maximum capacity.
  int z = 0; // Appears to be a running total of passengers.
  cin >> n;
  // A large, fixed-size array to store exit/enter pairs.
  int a[1000][2] = {0}; 
  
  /**
   * @brief Block to read all stop data into an array.
   */
  for (int i = 0; i < n; i++)
    cin >> a[i][0] >> a[i][1]; // a[i][0] = exits, a[i][1] = enters

  /**
   * @brief This block attempts to calculate the capacity but contains logical errors.
   *
   * @flaw 1: `z += a[i][1] - a[i + 1][0]` incorrectly mixes data from the current
   *          stop (`a[i]`) and the next stop (`a[i+1]`). When `i = n-1`, this
   *          results in an out-of-bounds read from `a[n][0]`.
   * @flaw 2: `a[i + 1][1] += z` nonsensically modifies the input data.
   * @flaw 3: `m = max(z, a[i][1])` incorrectly compares a running total with the
   *          number of passengers *entering* at stop `i`, not the total on board.
   */
  for (int i = 0; i < n; i++) {
    z += a[i][1] - a[i + 1][0];
    a[i + 1][1] += z;
    m = max(z, a[i][1]);
  }
  
  // @flaw 4: The program outputs `n`, the number of stops, which is not the problem's solution.
  cout << n; 
  
  return 0;
}
