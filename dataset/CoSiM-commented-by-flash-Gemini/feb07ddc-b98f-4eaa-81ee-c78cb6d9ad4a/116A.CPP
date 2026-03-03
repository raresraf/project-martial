/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 * This program calculates the minimum capacity a tram must have to carry all passengers
 * through a series of stops, where passengers exit and enter at each stop.
 * It tracks the current number of passengers after each stop and finds the peak occupancy.
 * Algorithm: Iterative simulation, storing passenger counts in an array and then finding the maximum.
 * Time Complexity: O(N) - where N is the number of tram stops. It involves two loops, each iterating up to N times.
 * Space Complexity: O(N) - an array of size N is used to store the passenger count after each stop.
 */
#include <iostream> // Required for standard input/output operations (cin, cout)
#include <algorithm> // Required for std::max

using namespace std; // Uses the standard namespace to avoid prefixing std::

long long d[11111]; // Global array to store the number of passengers in the tram after each stop.
                    // The size is chosen to accommodate up to 11110 stops, likely sufficient for competitive programming constraints.

int main() {
  int n; // Declares an integer to store the total number of tram stops.
  cin >> n; // Reads the total number of tram stops from standard input.
  
  int x, y; // x: passengers exiting, y: passengers entering.
  cin >> x >> y; // Reads exit and enter counts for the first stop.
  
  // Initializes the passenger count after the first stop. Assuming the tram starts empty,
  // the number of passengers after the first stop is simply 'y' (those who entered).
  // The 'x' (exiting passengers) for the first stop is implicitly handled as if no one was on board.
  d[0] = y;
  
  long long ans = 0; // Variable to store the maximum number of passengers found, initialized to 0.
  
  /**
   * Block Logic: Simulates passenger movements for the remaining tram stops (from second stop onwards).
   * This loop calculates the number of passengers in the tram after each subsequent stop.
   * Invariant: `d[i]` stores the number of passengers in the tram after the i-th stop.
   */
  for (int i = 1; i < n; ++i) { // Loop from the second stop (index 1) up to the last stop.
    cin >> x >> y; // Reads exit and enter counts for the current stop.
    // Calculates the current number of passengers by taking the previous count,
    // subtracting those who exited, and adding those who entered.
    d[i] = d[i - 1] + y - x;
  }
  
  /**
   * Block Logic: Determines the maximum number of passengers observed across all stops.
   * This loop iterates through the 'd' array to find the peak passenger count.
   * Invariant: 'ans' stores the maximum value encountered in 'd' so far.
   */
  for (int i = 0; i < n; ++i)
    ans = max(ans, d[i]); // Updates 'ans' if the current stop's passenger count is higher.
    
  // Prints the maximum capacity required for the tram.
  cout << ans;
  
  return 0; // Indicates successful program execution.
}