/**
 * @file 116A.CPP
 * @brief C-style solution for Codeforces problem 116A, "Tram".
 *
 * @description
 * This program calculates the minimum required capacity of a tram. It is given
 * the number of stops, and for each stop, the number of passengers exiting and
 * entering. The tram starts empty. The required capacity is the maximum number
 * of people on the tram at any single point in time.
 *
 * @algorithm
 * 1. Initialize `max` (for maximum capacity) and `seat` (for current occupancy) to 0.
 * 2. Read the number of stops, `n`.
 * 3. Loop `n` times to simulate each stop:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current occupancy: `seat = seat - a + b`.
 *    c. If the new occupancy `seat` is greater than `max`, update `max` to `seat`.
 * 4. After the loop, print the final value of `max`.
 *
 * @time_complexity O(N), where N is the number of stops.
 * @space_complexity O(1).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  // n: number of stops.
  // a: passengers exiting.
  // b: passengers entering.
  // max: tracks the maximum number of people on the tram at any time.
  // seat: tracks the current number of people on the tram.
  int n, a, b, max = 0, seat = 0;

  scanf("%d", &n);

  // Block Logic: Simulate the passenger flow for each of the 'n' stops.
  while (n--) {
    scanf("%d %d", &a, &b);
    // Update the current number of people on the tram.
    seat -= a;
    seat += b;
    // If the current occupancy is a new peak, update the max capacity required.
    if (seat > max)
      max = seat;
  }

  printf("%d\n", max);

  return 0;
}
