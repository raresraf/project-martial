/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program simulates the passenger flow on a tram across several stops.
 * It determines the maximum number of passengers inside the tram at any given
 * moment, which corresponds to the minimum capacity the tram must have.
 *
 * @algorithm Greedy Simulation
 * @complexity
 * Time: O(N), where N is the number of tram stops, as we iterate through
 *       the stops once.
 * Space: O(1), as we only use a few variables to store the current state.
 */
#include <iostream>
#include <algorithm> // For std::max

// Using the standard namespace for competitive programming convenience.
using namespace std;

/**
 * @brief Main function to read stop data and compute minimum capacity.
 */
int main() {
  // amt: total number of stops.
  // maxn: tracks the maximum number of passengers seen so far (the required capacity).
  // out, in: passengers exiting and entering at the current stop.
  // now: the current number of passengers on the tram.
  int amt, maxn = 0, i, out, in, now = 0;
  
  // Read the total number of stops.
  cin >> amt;
  
  // Pre-condition: 'now' is 0 (tram is empty initially). 'maxn' is 0.
  // Block Logic: Loop through each stop to simulate passenger changes.
  for (i = 1; i <= amt; i++) {
    // Read the number of passengers getting out and getting in at the current stop.
    cin >> out >> in;
    
    // Update the current number of passengers on the tram.
    now += in - out;
    
    // Update the maximum number of passengers if the current number is greater.
    maxn = max(maxn, now);
  }
  
  // Output the final calculated minimum capacity.
  cout << maxn << endl;
  
  return 0;
}
