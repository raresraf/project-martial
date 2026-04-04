/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem on platforms like Codeforces.
 * @details This program calculates the minimum required capacity of a tram by
 * tracking the number of passengers at each stop. It finds the maximum
 * number of passengers present on the tram at any single point in time.
 *
 * Algorithm:
 * 1. Read the number of tram stops, n.
 * 2. Initialize 'temp' (current passengers) and 'max' (maximum passengers) to 0.
 * 3. Loop 'n' times, once for each stop:
 *    a. Read the number of passengers exiting ('a') and entering ('b').
 *    b. Update the current passenger count: temp = temp - a + b.
 *    c. If the current count 'temp' is greater than the running 'max', update 'max'.
 * 4. After the loop, 'max' holds the highest passenger count, which is the minimum
 *    required capacity. Print 'max'.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1).
 */
#include <stdio.h>
// 'using namespace std;' is included but not strictly necessary as C-style I/O is used.
using namespace std;

int main() {
  int n, a, b, temp = 0, max = 0;
  scanf("%d", &n);

  /**
   * @brief Loop through each stop to calculate passenger changes.
   * Pre-condition: 'n' is the number of stops. 'temp' and 'max' are 0.
   * Invariant: After each iteration, 'temp' holds the current number of passengers
   * and 'max' holds the maximum number of passengers seen so far.
   */
  while (n--) {
    scanf("%d %d", &a, &b);
    // Update the current number of passengers by subtracting those who exit and adding those who enter.
    temp += b - a;
    // Update the maximum required capacity if the current number of passengers is greater.
    if (temp > max)
      max = temp;
  }
  // Output the minimum capacity required for the tram.
  printf("%d", max);
  return 0;
}
