/**
 * @file 282A.CPP
 * @brief Simulation of a minimalist stack-less register-based programming language.
 * 
 * Logic: Interprets a sequence of increment (++) and decrement (--) operations 
 * on a single global variable 'X', supporting both prefix and postfix notation.
 * 
 * Algorithm: Pattern matching on command strings to update a numeric accumulator.
 * Time Complexity: $O(N)$ where N is the number of instructions.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string s;
  int i, n, j, k, l;
  
  // Reads the total number of operations to execute.
  cin >> n;
  
  // Accumulator representing the current state of variable 'X'.
  j = 0;

  /**
   * Block Logic: Main instruction decoding loop.
   * Invariant: Evaluates each command independently and updates the state of 'j'.
   */
  for (i = 0; i < n; i++) {
    cin >> s;
    
    /**
     * Functional Utility: Maps linguistic patterns (strings) to arithmetic transitions.
     * Logic: Detects '++' or '--' by checking character positions (prefix/postfix support).
     */
    if (s[0] == 'X' && s[1] == '+') {
      j++; // Post-increment
    }
    if (s[0] == '+') {
      j++; // Pre-increment
    }
    if (s[0] == 'X' && s[1] == '-') {
      j--; // Post-decrement
    }
    if (s[0] == '-') {
      j--; // Pre-decrement
    }
  }
  
  // Output the final state of the virtual register.
  cout << j << endl;
  
  return 0;
}
