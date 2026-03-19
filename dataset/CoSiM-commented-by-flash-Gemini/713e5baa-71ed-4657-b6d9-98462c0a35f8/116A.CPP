/**
 * @file 116A.CPP
 * @brief Solution to Codeforces Problem 116A: "Tram".
 * Calculates the minimum capacity a tram needs to have, given a series of stops
 * where passengers get on and off. The capacity must be sufficient to hold the
 * maximum number of passengers in the tram at any single point.
 * Algorithm: Simulate the tram's journey, tracking the current number of passengers.
 * The maximum passenger count encountered during the journey determines the minimum capacity.
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, including iostream and algorithm for max.
using namespace std;

/**
 * @brief Main function to solve the "Tram" problem.
 * Reads the number of stops and for each stop, the number of passengers getting off
 * and on, then calculates and prints the maximum number of passengers at any time.
 */
int main() {
  int stops; // Stores the total number of tram stops.
  cin >> stops; // Reads the number of stops.
  
  int maximum = -1; // Stores the maximum number of passengers observed in the tram. Initialized to -1 to correctly handle the first `cur` calculation.
  int cur = 0;    // Stores the current number of passengers in the tram, initialized to 0.

  /**
   * Block Logic: Simulates the tram's journey through each stop.
   * Pre-condition: 'stops' is the total number of stops. 'cur' is 0, 'maximum' is -1.
   * Invariant: After each iteration, 'cur' reflects the passenger count after the current stop,
   *            and 'maximum' holds the highest passenger count seen so far.
   */
  while (stops--) { // Loop iterates 'stops' times, decrementing 'stops' with each iteration.
    int ups, downs; // 'downs': passengers getting off, 'ups': passengers getting on.
    cin >> downs >> ups; // Reads the number of passengers getting off and on at the current stop.
    
    cur = cur - downs + ups; // Updates the current number of passengers.
    maximum = max(maximum, cur); // Updates 'maximum' if 'cur' is greater than the current maximum.
  }
  
  cout << maximum; // Prints the final maximum number of passengers, which is the minimum required capacity.
  return 0;
}