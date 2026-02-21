/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity for a tram based on passenger flow.
 *
 * This program simulates passenger traffic on a tram across several stops.
 * For each stop, it's given the number of passengers exiting and entering.
 * It computes the maximum number of passengers present inside the tram at any
 * single time, which determines the minimum capacity the tram must have.
 *
 * Algorithm:
 * 1. Read the total number of stops, `n`.
 * 2. Initialize `x` to 0, representing the current number of passengers inside the tram.
 * 3. Initialize `capacity` to 0, which will store the maximum value `x` attains.
 * 4. Loop `n` times, once for each stop:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current passenger count: `x = x - a + b`.
 *    c. If the current passenger count `x` is greater than `capacity`, update `capacity` to `x`.
 * 5. After the loop, `capacity` holds the minimum required capacity. Print this value.
 *
 * Time Complexity: O(N), where N is the number of stops, due to the single loop.
 * Space Complexity: O(1), as only a few variables are used for tracking state.
 */
#include <cstdio>
using namespace std;

int main() {
  int n, a, b, x = 0, capacity = 0;
  // Read the number of stops.
  scanf("%d", &n);
  
  /**
   * @brief Loop through each stop to simulate passenger changes.
   * Pre-condition: `n` is the number of stops to simulate.
   * Invariant: At the start of each iteration, `x` holds the current passenger count,
   * and `capacity` holds the maximum count observed so far.
   */
  while (n--) {
    // Read passengers exiting (a) and entering (b).
    scanf("%d%d", &a, &b);
    // Update the number of passengers currently on the tram.
    x = x - a + b;
    
    // If the current passenger count exceeds the max capacity seen, update capacity.
    if (x > capacity)
      capacity = x;
  }
  
  // Print the calculated minimum capacity.
  printf("%d", capacity);
}
