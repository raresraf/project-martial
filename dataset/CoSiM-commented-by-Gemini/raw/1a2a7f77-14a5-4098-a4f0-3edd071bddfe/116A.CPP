/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem.
 * @details This program calculates the minimum possible capacity of a tram. It does this
 * by tracking the number of passengers inside the tram at each stop and finding the
 * maximum number of passengers that are ever on the tram at the same time.
 *
 * @section algorithm Algorithm
 * 1. Read the number of tram stops, `a`.
 * 2. Initialize `current_passengers` to 0.
 * 3. An array `passenger_counts` is used to store the number of passengers on the tram
 *    after each stop.
 * 4. The program loops `a` times, once for each stop.
 *    a. In each iteration, it reads the number of passengers exiting (`b`) and
 *       entering (`c`).
 *    b. It updates the `current_passengers` count: `current_passengers = current_passengers - b + c`.
 *    c. The new `current_passengers` count is stored in the `passenger_counts` array.
 * 5. After the loop, the `passenger_counts` array contains the passenger load after each stop.
 *    The required capacity is the maximum value in this array.
 * 6. The program sorts the array in ascending order.
 * 7. The last element of the sorted array is the maximum value, which is then printed as the answer.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(a log a), dominated by the sorting step. A more optimal O(a)
 *   solution is possible by tracking the maximum passenger count inside the loop,
 *   avoiding the need for an array and sorting.
 * - Space Complexity: O(a) to store the passenger counts at each stop in the array.
 */
#include <algorithm>
#include <iostream>
using namespace std;

/**
 * @brief Main function to calculate the minimum tram capacity.
 * @return 0 on successful execution.
 */
int main() {
  int num_stops;
  cin >> num_stops;
  
  int exiting, entering;
  int current_passengers = 0;
  // This array stores the passenger count after each stop.
  int passenger_counts[1000]; // Assuming max 1000 stops based on problem constraints.

  for (int i = 0; i < num_stops; i++) {
    cin >> exiting >> entering;
    // Update the number of passengers on the tram.
    current_passengers = current_passengers - exiting + entering;
    // Store the count for this stop.
    passenger_counts[i] = current_passengers;
  }
  
  // Sort the array of passenger counts to find the maximum.
  sort(passenger_counts, passenger_counts + num_stops);
  
  // The last element of the sorted array is the maximum passenger count.
  cout << passenger_counts[num_stops - 1];
  
  return 0;
}