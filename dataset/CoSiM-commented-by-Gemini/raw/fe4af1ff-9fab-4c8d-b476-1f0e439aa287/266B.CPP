/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem on platforms like Codeforces.
 * @details This program simulates a queue of children (Boys 'B' and Girls 'G').
 * Over a given time 't', any boy standing immediately in front of a girl swaps
 * places with her. This process is repeated for 't' seconds.
 *
 * Algorithm:
 * 1. Read the number of children 'n', the simulation time 't', and the initial queue 's'.
 * 2. Loop 't' times to simulate each second:
 *    a. Create a temporary empty string 'temp' to build the new queue state.
 *    b. Iterate through the current queue 's':
 *       i. If a 'B' is followed by a 'G' (s[i]=='B' and s[i+1]=='G'), append "GB"
 *          to 'temp' and skip the next character in 's' since it has been processed.
 *       ii. Otherwise, append the current character s[i] to 'temp'.
 *    c. Replace the old queue state 's' with the new state from 'temp'.
 *    d. Clear 'temp' for the next iteration.
 * 3. Print the final state of the queue 's'.
 *
 * Time Complexity: O(T * N), where T is the time and N is the number of children.
 * Space Complexity: O(N) to store the temporary string for the new queue state.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {

  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  int i;
  string temp;

  /**
   * @brief Main simulation loop, iterates for 't' seconds.
   * Pre-condition: 't' is the total simulation time. 's' holds the current queue state.
   * Invariant: After each outer loop iteration, 's' represents the queue state
   * after one second of swaps.
   */
  while (t--) {
    /**
     * @brief Inner loop to scan the queue and perform swaps for one time step.
     * Logic: This loop constructs a new string 'temp' representing the queue after
     * one second. It identifies adjacent 'BG' pairs and swaps them to 'GB' in the
     * new string.
     */
    for (i = 0; i < n; i++) {
      // Check for a Boy ('B') followed by a Girl ('G').
      if (s[i] == 'B' && s[i + 1] == 'G') {
        // Perform the swap in the temporary string.
        temp += "GB";
        // Increment 'i' to skip the next character, which has been processed as part of the pair.
        i++;
        continue;
      }
      // If no swap occurs, append the current character to the temporary string.
      temp += s[i];
    }
    // Update the queue to its new state for the next time step.
    s = temp;
    // Reset the temporary string.
    temp = "";
  }

  // Output the final arrangement of the queue.
  cout << s << endl;
}
