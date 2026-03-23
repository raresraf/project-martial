/**
 * @file 116A.CPP
 * @brief This competitive programming solution calculates the maximum capacity required
 *        for a tram/bus given a sequence of passenger exits and entries at various stops.
 *        It simulates the passenger flow and keeps track of the peak number of passengers.
 * Algorithm: Simulation. The code iteratively tracks the current number of passengers
 *            in the tram. For each stop, it subtracts exiting passengers and adds entering
 *            passengers, updating a running total. It continuously records the highest
 *            running total achieved, which represents the maximum capacity needed.
 * Time Complexity: O(N), where N is the number of stops, as the program processes
 *                  each stop with a constant number of operations.
 * Space Complexity: O(1), as only a few fixed-size integer variables are used.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std;

/**
 * @brief Main function of the program. Handles input, simulates passenger flow,
 *        and determines the maximum tram capacity.
 * Functional Utility: Serves as the entry point for execution, solving the tram capacity problem.
 */
int main() {
  int paradas;    // Represents the total number of stops.
  int bajan;      // Represents the number of passengers exiting at a stop.
  int suben;      // Represents the number of passengers entering at a stop.
  int p = 0;      // Current number of passengers in the tram; initialized to 0.
  int minimum = 0; // Tracks the maximum number of passengers encountered;
                   // named `minimum` but functions as a maximum accumulator.
  cin >> paradas; // Reads the total number of stops.

  // Block Logic: Iterates through each stop to simulate passenger movements and
  //              determine the peak passenger count.
  // Invariant: After each iteration, `p` holds the current passenger count, and
  //            `minimum` stores the highest `p` value recorded so far.
  for (int i = 0; i < paradas; i++) {
    cin >> bajan;   // Reads the number of passengers who exit at the current stop.
    p = p - bajan;  // Updates the current passenger count by subtracting exits.

    cin >> suben;   // Reads the number of passengers who enter at the current stop.
    p = p + suben;  // Updates the current passenger count by adding entries.
    // Block Logic: Updates `minimum` (which is tracking the maximum) if the current
    //              passenger count `p` is greater than the previously recorded maximum.
    if (p > minimum) {
      minimum = p; // Updates the maximum observed passenger count.
    }
  }

  cout << minimum << endl; // Prints the final maximum capacity required.

  return 0; // Indicates successful program termination.
}
