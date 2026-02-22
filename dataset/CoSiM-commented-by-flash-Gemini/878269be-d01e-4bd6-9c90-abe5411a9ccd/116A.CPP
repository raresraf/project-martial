/**
 * @878269be-d01e-4bd6-9c90-abe5411a9ccd/116A.CPP
 * @brief Solution for a competitive programming problem to find the maximum capacity needed for a tram.
 *
 * This program simulates a tram route with passengers exiting and entering at each stop.
 * The objective is to determine the maximum number of passengers present in the tram at any point
 * during its journey, which represents the minimum required capacity of the tram.
 *
 * Algorithm:
 * 1. Initialize current passengers ('sum') and maximum passengers observed ('max') to zero.
 * 2. Read the total number of tram stops 'n'.
 * 3. Iterate 'n' times, once for each stop:
 *    a. Read the number of passengers who exit ('a') and enter ('b') at the current stop.
 *    b. Update 'sum': first, subtract 'a' (exiting passengers), then add 'b' (entering passengers).
 *    c. If 'sum' (current passengers) is greater than 'max', update 'max' to 'sum'.
 * 4. After processing all stops, print the final 'max' value.
 *
 * Time Complexity: O(N) where N is the number of stops, as each stop involves constant time operations.
 * Space Complexity: O(1) as only a few variables are used to store counts.
 */

#include <stdio.h> // Functional Utility: Provides C-style input/output functions (scanf, printf).

int main() {
  int n, a, b, sum = 0, max = 0;
  // Functional Utility: Reads the total number of tram stops 'n'.
  scanf("%d", &n);

  // Block Logic: Loops 'n' times to process passenger changes at each stop.
  // Pre-condition: 'n' is a positive integer, followed by 'n' pairs of integers (a, b).
  // Invariant: 'sum' tracks current passengers, 'max' stores the peak count.
  for (int i = 1; i <= n; i++) {
    // Functional Utility: Reads the number of passengers exiting ('a') and entering ('b').
    scanf("%d %d", &a, &b);
    sum = sum - a; // Functional Utility: Passengers 'a' exit the tram.
    sum = sum + b; // Functional Utility: Passengers 'b' enter the tram.

    // Conditional Logic: Updates 'max' if the current number of passengers ('sum') is higher.
    if (sum > max)
      max = sum; // Functional Utility: Stores the new maximum passenger count.
  }
  // Functional Utility: Prints the maximum number of passengers observed, followed by a newline.
  printf("%d\n", max);
  return 0; // Functional Utility: Indicates successful program execution.
}
