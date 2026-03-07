/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem (Codeforces 266B).
 *
 * @details
 * This program simulates the reordering of a queue of boys 'B' and girls 'G'
 * over 't' seconds. In each second, any boy standing directly in front of a
 * girl swaps places with her.
 *
 * Algorithm:
 * 1. Read the number of children `n` and the time `t`.
 * 2. Read the initial queue configuration character by character into a string.
 *    (This is an unusual way to read the string for this problem).
 * 3. Loop `t` times to simulate each second.
 * 4. In each time step, iterate through the queue and swap any adjacent 'BG' pairs
 *    to 'GB'.
 * 5. After a swap at index `i`, the loop counter is incremented by an extra 1
 *    to ensure the boy who just moved does not move again in the same time step.
 * 6. Print the final state of the queue.
 *
 * Time Complexity: O(T * N), where T is the time and N is the queue length.
 * Space Complexity: O(N) to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  /**
   * @brief Reads the initial queue configuration character by character.
   * This is an unconventional and inefficient method for reading the input string
   * for this problem, which is typically read as a whole string.
   */
  while (n--) {
    char aux;
    cin >> aux;
    s.push_back(aux);
  }
  /**
   * @brief Outer loop to simulate 't' seconds of time.
   */
  while (t--) {
    /**
     * @brief Inner loop to perform all swaps for a single time step.
     * The loop iterates up to the second-to-last character to safely check pairs.
     */
    for (int i = 0; i < s.size() - 1; i++) {
      if (s[i] == 'B' and s[i + 1] == 'G') {
        // Swap the adjacent 'B' and 'G'.
        s[i] = 'G';
        s[i + 1] = 'B';
        // Increment 'i' to prevent the swapped boy from moving again in the same time step.
        i = i + 1;
      }
    }
  }
  cout << s << endl;
  return 0;
}
