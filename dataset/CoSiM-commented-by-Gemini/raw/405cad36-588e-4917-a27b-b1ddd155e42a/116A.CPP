/**
 * @file 116A.CPP
 * @brief Solution to Codeforces problem 116A - Tram.
 *
 * @algorithm
 * The problem asks for the minimum possible capacity of a tram, which is equivalent
 * to finding the maximum number of passengers inside the tram at any given moment.
 * The simulation proceeds through a series of stops.
 *
 * The algorithm is as follows:
 * 1. Initialize `max_capacity` and `current_passengers` to 0.
 * 2. Read the total number of stops, `n`.
 * 3. Loop `n` times, once for each stop.
 * 4. In each iteration, read the number of exiting passengers `a` and entering
 *    passengers `b`.
 * 5. Update the `current_passengers` by subtracting `a` and adding `b`.
 * 6. After the update, if `current_passengers` is greater than `max_capacity`,
 *    update `max_capacity` to the new `current_passengers` value.
 * 7. After the loop finishes, `max_capacity` will hold the maximum number of
 *    passengers that were ever on the tram simultaneously.
 * 8. Print `max_capacity`.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of stops, due to the single loop.
 *   The custom `readInt` function processes each digit, but this is still linear
 *   with the input size.
 * - Space Complexity: O(1), as only a few variables are used regardless of the
 *   number of stops.
 */
//
// Created by Taewoo Kim on 3/23/2017.
//

#include <iostream>

using namespace std;

/**
 * @brief Reads a positive integer from standard input character by character.
 *
 * This is a common competitive programming optimization for faster I/O compared
 * to std::cin. It skips any non-digit characters until it finds a digit, then
 * builds the integer until a non-digit character is found.
 * @return The parsed positive integer.
 */
int readInt() {
  char ch = getchar();
  while (true) {
    if (ch >= '0' && ch <= '9')
      break;
    ch = getchar();
  }
  int result = ch - '0';
  while (true) {
    ch = getchar();
    if (ch < '0' || ch > '9')
      break;
    result = result * 10 + ch - '0';
  }
  return result;
}

int main() {
  // find max people at any given time (people exit before others enter)
  int max = 0, count = 0;
  int n = readInt();
  /**
   * @brief Simulates the tram's journey through `n` stops.
   * Invariant: `count` holds the number of passengers after the events of the
   * current stop, and `max` holds the maximum value `count` has reached so far.
   */
  for (int i = 0; i < n; i++) {
    int a = readInt(); // Passengers exiting
    int b = readInt(); // Passengers entering
    // people get off first, then get on
    count += b - a;
    if (count > max)
      max = count;
  }

  cout << max;
  return 0;
}