/**
 * @89f445d2-11ce-485c-8978-a8f189a73ee0/266B.CPP
 * @brief This program simulates a queue reordering process where 'B' (boy) and 'G' (girl) elements
 * swap positions. Specifically, if a boy is followed by a girl, they swap. This process
 * repeats for a given number of time steps.
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the string.
 * Space Complexity: O(N) to store the character string.
 */

#include <iostream>
using namespace std;

int main() {
  // @var n: Represents the length of the queue (string).
  // @var t: Represents the number of time steps for the simulation.
  int n, t;
  cin >> n >> t;
  
  // @var str: The string representing the queue of 'B' and 'G' characters.
  string str;
  cin >> str;
  
  // Block Logic: Orchestrates the temporal progression of the simulation.
  // Invariant: At the start of each iteration, 't' represents the remaining time steps.
  // Pre-condition: 't' is initially the total number of simulation steps.
  while (t > 0) {
    // Block Logic: Performs a single-pass sweep to resolve 'B' followed by 'G' inversions.
    // Invariant: The loop iterates through most of the string to check for adjacent swaps.
    // Pre-condition: 'str' contains the current state of the queue.
    for (int i = 0; i < n - 1; i++) {
      // Block Logic: Checks for an adjacent 'B' and 'G' pair.
      // If a 'B' is followed by a 'G', they are swapped.
      if (str[i] == 'B' && str[i + 1] == 'G') {
        swap(str[i], str[i + 1]);
        // Inline: Increment 'i' to skip the next character, as it has already been moved.
        // This ensures that a single 'B' moves only one position per time step.
        i += 1; 
      }
    }
    // Inline: Decrement the time step counter.
    t -= 1;
  }
  
  // Post-condition: Output the final state of the queue after 't' time steps.
  cout << str << endl;
  return 0; // Explicitly return 0 for successful execution.
}
