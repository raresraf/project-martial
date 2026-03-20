/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem (Codeforces 266B).
 *
 * @details
 * This program simulates the arrangement of a queue of boys ('B') and girls ('G')
 * over a period of 't' seconds. In each second, any boy standing immediately
 * in front of a girl swaps places with her.
 *
 * @section algorithm Algorithm
 * The simulation uses a nested loop structure.
 * 1. The outer loop iterates 't' times, representing each second.
 * 2. The inner loop iterates through the queue from left to right.
 * 3. Inside the inner loop, if a 'B' is found at position `j` and a 'G' at `j+1`,
 *    they are swapped.
 * 4. A crucial step is to increment the inner loop's counter (`j++`) after a swap.
 *    This ensures that the boy who just moved does not move again in the same
 *    time step, correctly modeling the simultaneous nature of the swaps within one second.
 *
 * @section complexity Complexity
 * - Time Complexity: O(T * N), where T is the time in seconds and N is the number
 *   of children in the queue.
 * - Space Complexity: O(N) to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Swaps two characters by reference.
 * @param a The first character.
 * @param b The second character.
 */
void swap(char &a, char &b) {
  char temp;
  temp = a;
  a = b;
  b = temp;
}
int main() {
  int s, t, f = 0; // s: queue size, t: time, f: unused variable.
  cin >> s >> t;
  string str;
  cin >> str;

  /**
   * @brief Outer loop for time simulation.
   * This loop runs for 't' seconds.
   */
  for (int i = 0; i < t; ++i) {
    /**
     * @brief Inner loop to process the queue for one second.
     * It scans for adjacent 'BG' pairs and swaps them.
     */
    for (int j = 0; j < s; ++j) {
      // Check for a boy ('B') followed by a girl ('G').
      // Note: `j+1` might access out of bounds if `j == s-1`. This works
      // if `str[s]` is the null terminator `\0`, as `\0 == 'G'` is false.
      if (str[j] == 'B' && str[j + 1] == 'G') {
        swap(str[j], str[j + 1]);
        // Increment j to skip the new position of the boy ('B').
        // This ensures that each person moves at most once per time step.
        j++;
      }
    }
  }

  // Print the final arrangement of the queue.
  cout << str;
}
