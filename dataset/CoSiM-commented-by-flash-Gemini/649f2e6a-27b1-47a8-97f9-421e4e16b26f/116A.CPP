/**
 * @649f2e6a-27b1-47a8-97f9-421e4e16b26f/116A.CPP
 * @brief This program solves problem 116A, "Tram".
 * It simulates a tram's journey through 'n' stops. For each stop, it tracks
 * the number of passengers exiting and entering, calculates the current number
 * of passengers, and determines the maximum number of passengers observed
 * in the tram at any single moment.
 *
 * Algorithm: Linear simulation with passenger count tracking and maximum value update.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1) for storing variables.
 */
#include <iostream> // Includes the standard input/output stream library.

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to simulate a tram's passenger count and find the maximum.
 * Reads the number of stops, then for each stop, reads passengers exiting and entering,
 * updates current and maximum passenger counts.
 * @return 0 on successful execution.
 */
int main() {
  int stops = 0;   // Declares 'stops', representing the total number of tram stops.
  int enter = 0;   // Declares 'enter', for passengers entering at a stop.
  int exit = 0;    // Declares 'exit', for passengers exiting at a stop.
  int maximum = 0; // Initializes 'maximum' to 0, which will store the highest passenger count.
  int total = 0;   // Initializes 'total' to 0, which stores the current number of passengers in the tram.
  
  cin >> stops; // Reads the total number of stops from standard input.
  
  /**
   * Block Logic: Simulates the tram's journey through each stop.
   * For each stop, it updates the current number of passengers and tracks the maximum observed.
   * Invariant: 'maximum' stores the highest passenger count encountered up to the current stop.
   */
  for (int i = 0; i < stops; i++) { // Loops 'stops' times, once for each stop.
    cin >> exit; // Reads passengers exiting at the current stop.
    cin >> enter; // Reads passengers entering at the current stop.
    total = total + enter - exit; // Inline: Updates the current number of passengers in the tram.
    if (maximum < total) { // Inline: If the current total exceeds the recorded maximum.
      maximum = total; // Updates 'maximum' to the new highest count.
    }
  }
  cout << maximum; // Prints the maximum number of passengers observed.
  return 0;        // Indicates successful program termination.
}
