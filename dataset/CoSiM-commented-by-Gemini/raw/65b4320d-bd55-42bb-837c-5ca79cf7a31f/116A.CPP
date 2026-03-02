/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem on Codeforces (Problem 116A).
 * @details This program calculates the minimum possible capacity a tram could have to
 * accommodate all passengers throughout its route. It simulates the passenger flow
 * stop by stop, keeping track of the current number of people on the tram and
 * recording the maximum number of people present at any given time.
 *
 * Algorithm: Maximum Occupancy Tracking.
 * Time Complexity: O(N), where N is the number of tram stops, due to a single loop over the stops.
 * Space Complexity: O(1), as it uses a fixed number of variables to maintain state.
 */
#include <algorithm>
#include <stdio.h>

int main() {
  // `num_stops`: The total number of stops on the tram's route.
  // `max_people`: Tracks the peak number of people ever on the tram simultaneously. This will be the final answer.
  // `capacity`: The current number of people inside the tram.
  int num_stops, max_people{0}, capacity{0};
  scanf("%d", &num_stops);

  /**
   * @brief Block Logic: Simulates the passenger flow at each tram stop.
   * Invariant: At the beginning of each iteration, `capacity` holds the number of
   * passengers arriving at the current stop, and `max_people` holds the maximum
   * capacity required for all previous stops.
   */
  for (int num_stop = 0; num_stop < num_stops; ++num_stop) {
    int leave, enter;
    // Reads the number of passengers leaving and entering at the current stop.
    scanf("%d%d", &leave, &enter);
    
    // Updates the current number of passengers. The change is the number of
    // people entering minus the number of people leaving.
    capacity = capacity + enter - leave;

    // Pre-condition: Checks if the new passenger count exceeds the maximum recorded so far.
    // `std::max` updates `max_people` to the new capacity if it's higher.
    max_people = std::max(max_people, capacity);
  }

  // After iterating through all stops, `max_people` holds the minimum required capacity.
  printf("%d", max_people);
  return 0;
}
