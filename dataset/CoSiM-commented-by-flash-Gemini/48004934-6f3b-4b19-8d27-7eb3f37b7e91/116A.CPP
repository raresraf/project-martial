/**
 * @48004934-6f3b-4b19-8d27-7eb3f37b7e91/116A.CPP
 * @brief Solution for a competitive programming problem to find the maximum capacity needed for a tram.
 *
 * This program simulates a tram route with passengers exiting and entering at each stop.
 * The objective is to determine the maximum number of passengers present in the tram at any point
 * during its journey, which represents the minimum required capacity of the tram.
 * This version uses standard C++ I/O streams (`iostream`).
 *
 * Algorithm:
 * 1. Read the total number of tram stops 'number'.
 * 2. Declare arrays to store the number of passengers exiting ('out') and entering ('in') at each stop.
 * 3. Initialize current passengers ('total') and maximum passengers observed ('minCap') to zero.
 * 4. In a first loop, read all 'out' and 'in' values for all stops.
 * 5. In a second loop, iterate through each stop:
 *    a. Update 'total' by adding 'in[i]' (entering passengers) and subtracting 'out[i]' (exiting passengers).
 *    b. If 'total' (current passengers) is greater than 'minCap', update 'minCap' to 'total'.
 * 6. After processing all stops, print the final 'minCap' value.
 *
 * Time Complexity: O(N) where N is the number of stops, as it involves two passes through the stops.
 * Space Complexity: O(N) to store the 'in' and 'out' passenger counts for all stops.
 */

#include <iostream> // Functional Utility: Provides standard input/output streams (std::cin, std::cout).

using namespace std;

int main() {
  int number;
  // Functional Utility: Reads the total number of tram stops.
  cin >> number;
  // Functional Utility: Declares arrays to store passengers exiting ('out') and entering ('in') at each stop.
  // 'total' tracks current passengers, 'minCap' stores the maximum observed passenger count.
  int in[number], out[number], i, total = 0, minCap = 0;

  // Block Logic: Reads the number of passengers exiting and entering for each stop.
  // Pre-condition: 'number' pairs of integers are available in the input.
  // Invariant: 'out' and 'in' arrays are populated with passenger data.
  for (i = 0; i < number; i++) {
    cin >> out[i] >> in[i];
  }

  // Block Logic: Iterates through each stop to calculate the maximum number of passengers.
  // Pre-condition: 'in' and 'out' arrays are populated.
  // Invariant: 'total' reflects current passengers, 'minCap' tracks the highest passenger count.
  for (i = 0; i < number; i++) {
    total = total + in[i] - out[i]; // Functional Utility: Updates the current passenger count.
    // Conditional Logic: Checks if the current passenger count exceeds the recorded maximum capacity.
    if (total > minCap)
      minCap = total; // Functional Utility: Updates the maximum capacity if a new peak is reached.
  }

  // Functional Utility: Prints the minimum required tram capacity.
  cout << minCap;
  return 0; // Functional Utility: Indicates successful program execution.
}
