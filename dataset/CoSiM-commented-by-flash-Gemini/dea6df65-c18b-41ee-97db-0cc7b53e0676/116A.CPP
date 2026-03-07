/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" competitive programming problem (Codeforces 116A).
 * The program simulates a tram journey with multiple stops. At each stop,
 * some passengers exit and others enter. The goal is to determine the minimum
 * capacity the tram must have to accommodate all passengers at any point
 * during its journey.
 * Algorithm: Iterative tracking of current passenger count and maximum seen count.
 * Time Complexity: O(N), where N is the number of tram stops.
 * Space Complexity: O(N) for storing passengers exiting and entering at each stop.
 */
#include <iostream> // For input/output operations.

using namespace std;

int main() {
  // Declares 'number' to store the total number of tram stops.
  int number;
  // Reads the total number of tram stops.
  cin >> number;
  
  // Declares arrays 'in' and 'out' of size 'number' to store the number of
  // passengers entering and exiting at each stop, respectively.
  int in[number], out[number], i;
  // 'total' keeps track of the current number of passengers in the tram.
  int total = 0;
  // 'minCap' stores the maximum number of passengers observed at any point,
  // which will be the minimum required capacity.
  int minCap = 0;

  /**
   * Block Logic: Reads the number of passengers exiting and entering at each stop.
   * Invariant: At the start of each iteration, 'i' is the current stop index.
   * Pre-condition: 'number' is a positive integer.
   * Post-condition: 'out' and 'in' arrays are populated with passenger counts.
   */
  for (i = 0; i < number; i++) {
    // Reads the number of passengers exiting and entering for the current stop.
    cin >> out[i] >> in[i];
  }

  /**
   * Block Logic: Simulates the tram journey to find the maximum passenger count.
   * Invariant: At the start of each iteration, 'i' is the current stop index,
   * 'total' is the passenger count after the previous stop, and 'minCap' is the
   * maximum passenger count observed up to the previous stop.
   * Pre-condition: 'in' and 'out' arrays are populated.
   * Post-condition: 'minCap' holds the overall maximum passenger count during the journey.
   */
  for (i = 0; i < number; i++) {
    // Updates the current total number of passengers in the tram.
    // Passengers exiting are subtracted, and passengers entering are added.
    total = total + in[i] - out[i];
    
    // Conditional Logic: Updates 'minCap' if the current total passenger count is higher.
    if (total > minCap)
      minCap = total;
  }

  // Prints the minimum required tram capacity.
  cout << minCap;
  
  return 0; // Indicates successful program execution.
}