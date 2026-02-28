/**
 * @file 116A.CPP
 * @brief Simulates and tracks the maximum occupancy of a facility over time.
 *
 * This program processes a series of entry and exit events for a facility.
 * For each event, it reads the number of people exiting and the number of people
 * entering. It's explicitly stated that people exit before new people enter
 * in each step. The goal is to determine the maximum number of people
 * present in the facility at any given moment. A custom `readInt` function
 * is used for efficient input reading.
 *
 * Algorithm: Iterative simulation of occupancy changes with fast input.
 * Time Complexity: O(N * D), where N is the number of entry/exit event pairs
 *                  and D is the average number of digits in the input integers.
 *                  The `readInt` function reads character by character, making
 *                  its complexity proportional to the number of digits.
 * Space Complexity: O(1) as only a few scalar variables are used to store
 *                   current count, maximum count, and loop indices.
 */

//
// Created by Taewoo Kim on 3/23/2017.
//

#include <iostream>

using namespace std;

// only positive numbers
// only positive numbers
/**
 * @brief Custom function to efficiently read positive integers from standard input.
 *
 * This function reads characters one by one using `getchar()` until a digit is encountered.
 * It then accumulates digits to form the integer until a non-digit character is met.
 * This method is generally faster than `std::cin` for competitive programming due
 * to reduced overhead.
 * @return The positive integer read from input.
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
  /**
   * @brief Main function to simulate facility occupancy and find maximum capacity.
   *
   * Initializes occupancy count and maximum recorded occupancy. Reads the number
   * of events and then processes each event (people exiting, then people entering)
   * to update the current count and track the peak.
   */
  // @param max Stores the maximum number of people observed in the facility.
  // @param count Stores the current number of people in the facility.
  int max = 0, count = 0;
  // @param n The total number of entry/exit events to process.
  int n = readInt();
  /**
   * @brief Loop to process each entry/exit event.
   * Invariant: At the start of each iteration, `count` reflects the number of people
   *            after processing `i` events, and `max` holds the peak occupancy so far.
   */
  for (int i = 0; i < n; i++) {
    // @param a Number of people exiting the facility.
    int a = readInt();
    // @param b Number of people entering the facility.
    int b = readInt();
    // people get off first, then get on
    /**
     * @brief Updates the current occupancy count.
     * Logic: People exiting (`a`) are subtracted from the count, and then
     *        people entering (`b`) are added. This order is crucial.
     */
    count += b - a;
    /**
     * @brief Updates the maximum occupancy if the current count exceeds it.
     */
    if (count > max)
      max = count;
  }

  // @brief Prints the overall maximum observed occupancy.
  cout << max;
  return 0;
}