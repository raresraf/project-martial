/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem on Codeforces.
 *
 * @details This program calculates the minimum required capacity of a tram.
 * It first reads and stores all the passenger exit and entry data for every
 * stop into a 2D array. It then processes this array to find the maximum
 * number of passengers on the tram at any point.
 *
 * Algorithm:
 * 1. Read the number of stops, n.
 * 2. Create a 2D array `stop[n][2]` to hold all exit/entry data.
 * 3. Populate the array by reading all passenger data from the input.
 * 4. Initialize `intram` (current passengers) and `max` (maximum so far) to 0.
 * 5. Iterate from the first to the last stop:
 *    a. Update the current passenger count using the stored exit and entry values.
 *    b. If the current count `intram` is greater than `max`, update `max`.
 * 6. Print the final `max` value.
 *
 * Time Complexity: O(N) for reading and processing.
 * Space Complexity: O(N) due to storing all stop data in a 2D array.
 */
#include <iostream>
using namespace std;

int main() {
  int n, intram = 0, max = 0;
  cin >> n;

  // Uses a Variable-Length Array (VLA) to store all stop data, a non-standard C++ feature.
  int stop[n][2];
  // Loop to read all exit and entry data into the 2D array first.
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 2; j++)
      cin >> stop[i][j];

  /**
   * @brief Loop through the stored data to find the maximum passenger count.
   * Pre-condition: `stop` array is filled with passenger data. `intram` and `max` are 0.
   * Invariant: After each iteration, `intram` holds the current passenger count,
   * and `max` holds the peak passenger count observed so far.
   */
  for (int i = 0; i < n; i++) {
    // Update current passenger count: subtract exits, add entries.
    intram -= stop[i][0];
    intram += stop[i][1];
    // If the current count is a new maximum, record it.
    if (intram > max)
      max = intram;
  }

  cout << max << endl;

  return 0;
}