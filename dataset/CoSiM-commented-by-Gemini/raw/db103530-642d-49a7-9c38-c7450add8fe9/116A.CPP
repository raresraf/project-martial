/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 *
 * @details
 * This program calculates the minimum required capacity of a tram based on the number
 * of passengers exiting and entering at each stop. It finds the maximum number of
 * people inside the tram at any given time during its journey.
 *
 * Algorithm:
 * 1. Read the total number of tram stops (n).
 * 2. Simulate the tram's journey stop by stop:
 *    a. For each stop, read the number of exiting (a) and entering (b) passengers.
 *    b. Update the current number of passengers in the tram.
 *    c. Store the passenger count after each stop in an array.
 * 3. Iterate through the array of passenger counts to find the maximum value.
 * 4. Print the maximum value, which represents the required tram capacity.
 *
 * Time Complexity: O(N), where N is the number of stops, due to two separate linear passes.
 * Space Complexity: O(N), to store the passenger count at each stop in an array.
 */
#include <iostream>
using namespace std;

// n: total number of tram stops
// a: number of passengers exiting at a stop
// b: number of passengers entering at a stop
int n, a, b;

int main() {
  // Read the number of stops.
  cin >> n;
  
  // VLA to store passenger count after each stop. Note: VLAs are a non-standard C++ extension.
  int arr[n];
  // 'init' tracks the current number of passengers inside the tram.
  int init = 0;

  /**
   * Block Logic: Simulates the passenger flow at each stop.
   * This loop iterates through 'n' stops, updating the passenger count
   * and recording the count at each step into the 'arr' array.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    // Update current passenger count: subtract exiting, add entering.
    init = init - a + b;
    arr[i] = init;
  }

  // 'm' will store the maximum passenger count found, representing the minimum required capacity.
  int m = 0;

  /**
   * Block Logic: Finds the maximum value in the passenger count array.
   * This loop iterates through the recorded passenger counts to determine the
   * peak number of passengers on the tram at any point.
   */
  for (int i = 0; i < n; i++) {
    // Invariant: If the current stop's passenger count is greater than the max
    // seen so far, update the max.
    if (arr[i] > m)
      m = arr[i];
  }

  // Output the minimum required capacity.
  cout << m << endl;

  return 0;
}
