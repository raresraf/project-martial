/**
 * @file 116A.CPP
 * @brief Solution for Codeforces Problem 116A - "Tram".
 * This program calculates the minimum capacity a tram must have to carry all passengers
 * through a series of stops, where passengers exit and enter at each stop.
 * It tracks the current number of passengers and finds the peak occupancy.
 * Algorithm: Iterative simulation, tracking current and maximum passenger counts.
 * Time Complexity: O(N) - where N is the number of tram stops/events.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 *
 * Note on loop: The loop iterates `n + 1` times. If 'n' represents the count of actual
 * stops, this implies an extra input pair is processed, or 'n' refers to the total
 * number of events including an initial implicit state.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream, algorithm for max).
using namespace std;     // Uses the standard namespace.

int main() {
  int n;     // Declares an integer to store the total number of stops/events.
  cin >> n;  // Reads the total number of stops/events from standard input.
  
  int a, b;  // a: passengers exiting, b: passengers entering.
  int maxn = 0; // Stores the maximum number of passengers observed in the tram (required capacity).
  int total = 0; // Stores the current number of passengers in the tram.
  
  /**
   * Block Logic: Simulates passenger movements at each tram stop/event.
   * The loop runs `n + 1` times, processing pairs of passengers exiting and entering.
   * `total` is updated with the current passenger count, and `maxn` tracks the highest `total` reached.
   * Invariant: `total` holds the current passenger count, `maxn` holds the peak passenger count.
   */
  for (int i = 0; i <= n; i++) { // Loop `n + 1` times to process all events.
    cin >> a >> b; // Reads passengers exiting ('a') and entering ('b').
    total -= a;    // Subtracts exiting passengers.
    total += b;    // Adds entering passengers.
    
    // Updates 'maxn' if the current passenger count 'total' is higher than the previously recorded maximum.
    maxn = max(total, maxn);
  }
  
  // Prints the maximum capacity required for the tram.
  cout << maxn << endl;
  
  return 0; // Indicates successful program execution.
}
