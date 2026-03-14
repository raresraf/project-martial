/**
 * @6915a6b3-a20e-46ea-8038-13f0b91c8075/116A.CPP
 * @brief This program solves Codeforces problem 116A, "Tram".
 * It simulates a tram's journey through a series of stops. For each stop,
 * it tracks the number of passengers exiting and entering, calculates the
 * current number of passengers on board, and continuously updates the
 * maximum number of passengers observed at any point during the journey.
 *
 * Algorithm: Linear simulation. It iteratively updates the current passenger count
 * after each exit and entry, and continuously tracks the maximum observed count.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1) for storing variables.
 */
#include <iostream> // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to simulate a tram's passenger count and find the maximum.
 * Reads the number of stops, then for each stop, reads passengers exiting and entering,
 * updates current and maximum passenger counts.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Declares 'n', representing the total number of tram stops.
  int i; // Loop counter.
  int a, b; // 'a' for passengers exiting, 'b' for passengers entering at a stop.
  int c = 0; // 'c' stores the current number of passengers in the tram.
  int m = 0; // 'm' stores the maximum number of passengers observed.
  
  cin >> n; // Reads the total number of stops from standard input.
  
  /**
   * Block Logic: Simulates the tram's journey through each of the 'n' stops.
   * For each stop, it reads passengers exiting and entering, updates the current
   * passenger count (`c`), and then updates the maximum observed (`m`)
   * if the current count is higher.
   * Invariant: `m` correctly holds the highest passenger count encountered so far.
   */
  for (i = 1; i <= n; i++) { // Loops `n` times, once for each stop.
    cin >> a >> b; // Reads passengers exiting ('a') and entering ('b') for the current stop.
    c = c - a + b; // Inline: Updates the current number of passengers in the tram.
    
    // Inline: If the current passenger count (`c`) is greater than the recorded maximum (`m`).
    if (c > m) 
      m = c; // Updates 'm' to the new highest count.
  }
  cout << m; // Prints the maximum number of passengers observed.
  return 0; // Indicates successful program termination.
}
