/**
 * @65b4320d-bd55-42bb-837c-5ca79cf7a31f/116A.CPP
 * @brief This program solves problem 116A, "Tram".
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
#include <algorithm> // For std::max.
#include <stdio.h>   // For C-style input/output (scanf, printf).

/**
 * @brief Main function to simulate a tram's passenger count and find the maximum.
 * Reads the number of stops, then for each stop, reads passengers exiting and entering,
 * updates current and maximum passenger counts.
 * @return 0 on successful execution.
 */
int main() {
  int num_stops;     // Declares 'num_stops', representing the total number of tram stops.
  int max_people{0}; // Initializes 'max_people' to 0, which will store the highest passenger count observed.
  int capacity{0};   // Initializes 'capacity' to 0, which stores the current number of passengers in the tram.
  
  scanf("%d", &num_stops); // Reads the total number of stops from standard input.
  
  /**
   * Block Logic: Simulates the tram's journey through each of the `num_stops`.
   * For each stop, it reads passengers exiting and entering, updates the current
   * passenger count (`capacity`), and then updates the maximum observed (`max_people`)
   * if the current count is higher.
   * Invariant: `max_people` correctly holds the highest passenger count encountered so far.
   */
  for (int num_stop = 0; num_stop < num_stops; ++num_stop) { // Loops `num_stops` times, once for each stop.
    int leave, enter; // Declares 'leave' for passengers exiting and 'enter' for passengers entering.
    scanf("%d%d", &leave, &enter); // Reads passengers exiting and entering for the current stop.
    capacity = capacity + enter - leave; // Inline: Updates the current number of passengers in the tram.
    max_people = std::max(max_people, capacity); // Inline: Updates 'max_people' to the new highest count using std::max.
  }
  printf("%d", max_people); // Prints the maximum number of passengers observed.
  return 0; // Indicates successful program termination.
}
