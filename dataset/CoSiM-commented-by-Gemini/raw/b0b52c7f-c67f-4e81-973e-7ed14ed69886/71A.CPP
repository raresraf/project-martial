/**
 * @file 71A.CPP
 * @brief Calculates total time for tasks on a circular track.
 * @details This program simulates movement on a circular track with positions numbered 1 to `n`.
 * It calculates the total time (or distance) required to complete `m` tasks, where each task
 * is located at a specific position on the track. Movement is assumed to be only in the
 * forward direction (e.g., from 1 to 2, ... to n, then wrapping around to 1).
 *
 * @algorithm
 * 1. Reads the size of the track `n` and the number of tasks `m`.
 * 2. Initializes the current position `pos` to 1 and total time `res` to 0.
 * 3. Loops `m` times, once for each task.
 * 4. In each loop, reads the next task's position `num`.
 * 5. Calculates the forward distance from `pos` to `num`:
 *    - If `num` is ahead of `pos`, the time is `num - pos`.
 *    - If `num` is behind `pos`, the movement wraps around the end of the track. The time is `(n - pos) + num`.
 * 6. Adds this time to the total `res` and updates `pos` to `num`.
 * 7. After the loop, prints the total accumulated time `res`.
 *
 * @complexity
 * - Time: O(M), where M is the number of tasks, as each task is processed in constant time.
 * - Space: O(1), as memory usage is constant.
 */
#include <iostream>

using namespace std;

int main() {
  // n: The number of positions on the circular track (e.g., houses on a street).
  // m: The number of tasks to be completed.
  // pos: The current position on the track, initialized to the start (position 1).
  int n, m, pos = 1;
  cin >> n >> m;

  // res: The total time/distance accumulated, using a 64-bit integer to prevent overflow.
  int64_t res = 0;
  
  /**
   * @brief Main loop to process all `m` tasks.
   * @invariant At the start of each iteration, `pos` is the location of the previous task,
   * and `res` is the total time spent up to that point.
   */
  while (m--) {
    int num; // The position of the current task.
    cin >> num;

    /**
     * @brief Conditional logic to calculate movement time.
     * It computes the time taken to move from the current position `pos` to the new task's position `num`,
     * assuming movement is always in the forward direction.
     */
    if ((num <= n) && num > pos) {
      // Case 1: Forward movement without wrap-around.
      res += num - pos;
      pos = num;
    } else if (num < pos) {
      // Case 2: Forward movement with wrap-around (e.g., from position 5 to 2 on a track of size 5).
      // The time is the distance to the end of the track plus the distance from the start.
      res += (n - pos) + num;
      pos = num;
    }
    // Case 3: if num == pos, no movement is needed, time taken is 0. This is implicitly handled.
  }

  // Output the total accumulated time.
  cout << res;

  return 0;
}