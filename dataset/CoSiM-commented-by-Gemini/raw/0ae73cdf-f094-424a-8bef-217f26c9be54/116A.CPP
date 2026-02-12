/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from competitive programming.
 * @details This program calculates the minimum required capacity of a tram. It simulates the
 * change in the number of passengers over `n` stops. At each stop, a certain number of
 * passengers exit and a new set of passengers enter. The program keeps track of the
 * current number of people inside the tram and updates a "maximum capacity" variable
 * whenever the current number of people exceeds the previously recorded maximum.
 *
 * @author Ikki
 * @date May 31, 2015
 *
 * @algorithm Simulation of passenger flow to find maximum occupancy.
 * @complexity
 * Time complexity: O(N), where N is the number of tram stops, as it processes each stop once.
 * Space complexity: O(1), using only a few variables to track state.
 */

#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  // n: the number of tram stops.
  int n;
  cin >> n;
  
  // val: the current number of people inside the tram.
  // ans: the maximum number of people recorded at any point, representing the required capacity.
  int val = 0, ans = 0;
  
  /**
   * @brief Main loop to simulate the tram's journey.
   * @invariant Iterates `n` times, once for each stop.
   */
  for (int i = 0; i < n; i++) {
    // a: number of passengers exiting.
    // b: number of passengers entering.
    int a, b;
    cin >> a >> b;
    
    // Update the current number of passengers.
    val -= a;
    val += b;
    
    // Update the maximum capacity if the current number of passengers is greater.
    ans = max(ans, val);
  }
  
  // Print the minimum required capacity.
  cout << ans << endl;
  return 0;
}
