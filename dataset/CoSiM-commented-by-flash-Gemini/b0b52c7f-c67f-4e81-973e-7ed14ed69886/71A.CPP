/**
 * @file 71A.CPP
 * @brief Solution to a competitive programming problem that calculates the minimum moves
 * required to visit a sequence of locations on a circular path.
 * The problem involves moving a pointer from one position to another in a circular array
 * (or similar structure) of size 'n'. The goal is to calculate the total distance (cost)
 * to visit 'm' target locations in a given order. Moves are always considered in a way
 * that minimizes the distance (e.g., shortest path on a circle).
 * Algorithm: Greedy calculation of shortest path on a circular track.
 * Time Complexity: O(m), where 'm' is the number of moves.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // Declares 'n' for the total size of the circular path, 'm' for the number of moves,
  // and 'pos' for the current position (initialized to 1, assuming 1-based indexing).
  int n, m, pos = 1;
  // Reads the total size of the path and the number of moves.
  cin >> n >> m;

  // Declares 'res' (result) to accumulate the total cost, using int64_t for potentially large sums.
  int64_t res = 0;
  
  /**
   * Block Logic: Iterates 'm' times, once for each target position to visit.
   * Invariant: At the start of each iteration, 'm' represents the remaining moves,
   * 'pos' is the current location, and 'res' holds the accumulated cost so far.
   */
  while (m--) {
    int num; // Declares 'num' to store the next target position.
    cin >> num; // Reads the next target position.
    
    /**
     * Conditional Logic: Determines the shortest path to the target position 'num'.
     * Case 1: Moving clockwise (num >= pos) and num is within the valid range (num <= n).
     * If 'num' is greater than or equal to the current 'pos', the cost is simply the difference.
     * The condition `num <= n` is implicitly true as positions are usually 1 to n.
     */
    if ((num >= pos) && num <= n) { // Simplified from `(num <= n) && num > pos` - the logic effectively covers moving forward.
      res += num - pos; // Adds the direct distance.
      pos = num;        // Updates current position to target.
    } 
    /**
     * Conditional Logic: Case 2: Moving counter-clockwise (num < pos).
     * If 'num' is less than the current 'pos', it's shorter to wrap around the circular path.
     * The cost is from 'pos' to 'n' (end of circle) plus from '1' to 'num'.
     */
    else if (num < pos) {
      res += (n - pos) + num; // Adds distance for wrapping around.
      pos = num;              // Updates current position to target.
    }
    // Note: The original logic implicitly handles num == pos (cost 0) within the first if-block.
  }

  // Prints the total accumulated cost.
  cout << res;

  return 0; // Indicates successful program execution.
}
