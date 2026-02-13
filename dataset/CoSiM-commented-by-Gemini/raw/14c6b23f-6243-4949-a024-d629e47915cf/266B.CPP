/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem.
 * @details This program simulates a queue of children ('B' for boy, 'G' for girl).
 * Over a given time `t`, any boy standing immediately in front of a girl will swap
 * places with her. This process is repeated for `t` seconds.
 *
 * @section algorithm Algorithm
 * The program performs a direct simulation of the process.
 * 1. Read the number of children `n`, the total time `t`, and the initial queue string `s`.
 * 2. An outer loop runs for `t` iterations, each representing one second of time.
 * 3. An inner loop iterates through the queue from left to right to perform all possible swaps for that second.
 *    - It checks for adjacent pairs of 'B' followed by 'G'.
 *    - When a "BG" pair is found, the characters are swapped to "GB".
 *    - After a swap at index `i`, the inner loop's index `i` is advanced by 2. This is a key step to ensure that
 *      a single person does not move more than one position in a single time step and to avoid re-checking the swapped pair.
 *    - If no swap occurs at index `i`, the index is advanced by 1.
 * 4. After `t` seconds, the final arrangement of the queue is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(t * n), where `t` is the time and `n` is the length of the queue.
 * - Space Complexity: O(n), to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  // This outer loop simulates the passage of time for 't' seconds.
  while (t--) {
    int i = 0;
    // This inner loop performs one pass of swaps over the queue for a single second.
    while (i < n - 1) {
      // If a boy ('B') is standing in front of a girl ('G')...
      if (s[i] == 'B' && s[i + 1] == 'G') {
        // ...they swap places.
        s[i] = 'G';
        s[i + 1] = 'B';
        // The index is advanced by 2 to prevent the just-moved boy from moving again
        // in the same time step.
        i = i + 2;
      } else {
        // If no swap occurs, simply move to the next position.
        i++;
      }
    }
  }
  // After 't' seconds, print the final state of the queue.
  cout << s;
}