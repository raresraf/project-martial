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
 * Space Complexity: O(N), for storing the `trum` (stop data) array.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std;

/**
 * @brief Defines a structure to hold passenger data for a single tram stop.
 */
typedef struct {
  int entry; // Number of passengers entering the tram at this stop.
  int exit;  // Number of passengers exiting the tram at this stop.
} trum; // Alias for the structure, representing a single stop's data.

/**
 * @brief Calculates the maximum number of passengers present in the tram at any point.
 * @param data A pointer to an array of `trum` structs, each representing a stop's data.
 * @param length The number of stops in the `data` array.
 * @return The maximum number of passengers observed in the tram during the entire journey.
 * Functional Utility: Encapsulates the core logic for simulating passenger flow and
 *                     determining the peak capacity requirement.
 */
int max_passengers(trum *data, int length) {
  int last_stop_passengers = 0; // Current number of passengers in the tram.
  int max_passengers = 0;       // Maximum number of passengers observed so far.

  // Block Logic: Iterates through each stop to simulate passenger movements and
  //              determine the peak passenger count.
  // Invariant: After each iteration, `last_stop_passengers` holds the current
  //            passenger count, and `max_passengers` stores the highest
  //            `last_stop_passengers` value recorded up to the current stop.
  for (int i = 0; i < length; i++) {
    // Updates the current passenger count: subtracts exits, adds entries.
    last_stop_passengers =
        (last_stop_passengers - data[i].exit) + data[i].entry;

    // Block Logic: Updates `max_passengers` if `last_stop_passengers` exceeds
    //              the currently recorded maximum.
    if (max_passengers < last_stop_passengers)
      max_passengers = last_stop_passengers;
  }

  return max_passengers; // Returns the overall maximum capacity required.
}

/**
 * @brief Main function of the program. Handles input, creates data structures,
 *        calls the `max_passengers` function, and prints the result.
 * Functional Utility: Serves as the entry point for execution, preparing data
 *                     and initiating the tram capacity calculation.
 */
int main() {
  int n;      // Represents the total number of stops.
  cin >> n;   // Reads the total number of stops.
  trum data[n]; // Declares a C-style array of `trum` structs to store stop data.
                // Note: Variable-length arrays (VLAs) are a C99 feature and
                // a GCC extension in C++; use std::vector for C++ standard compliance.

  // Block Logic: Reads the passenger exit and entry counts for each stop and
  //              populates the `data` array of `trum` structs.
  // Invariant: After this loop, the `data` array contains all stop details.
  for (int i = 0; i < n; i++)
    cin >> data[i].exit >> data[i].entry; // Reads exit and entry for each stop.

  // Calls `max_passengers` to calculate the peak capacity and prints the result.
  cout << max_passengers(data, n) << endl;

  return 0; // Indicates successful program termination.
}