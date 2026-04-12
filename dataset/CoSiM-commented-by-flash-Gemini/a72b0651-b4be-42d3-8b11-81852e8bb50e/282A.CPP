/**
 * @file 282A.CPP
 * @brief Simulation of a register-based state machine for a minimalist language.
 * 
 * Logic: Tracks the state of a single numeric register 'X' across a series of 
 * increment (++) and decrement (--) operations. The language supports both 
 * prefix and postfix operators.
 * 
 * Algorithm: Pattern-based string matching to trigger state transitions.
 * Time Complexity: $O(N \times L)$ where N is the operation count and L is the command length.
 * Space Complexity: $O(N \times L)$ to buffer the entire instruction sequence.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

using namespace std;

#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define REV(i, b, a) for (int i = (a); i >= (b); i--)
#define mp make_pair
#define pb push_back

int main(int argc, char *argv[]) {
  // Functional Utility: Reads total instruction count.
  unsigned short n;
  // State: Accumulator register representing 'X'.
  signed short X = 0;
  string s;
  // Data Structure: Buffers all input commands before processing.
  vector<string> input;
  
  cin >> n;
  FOR(i, 0, n) {
    cin >> s;
    input.pb(s);
  }

  /**
   * Block Logic: Sequentially interprets the buffered instructions.
   * Invariant: Updates state 'X' based on the presence of operational substrings.
   */
  FOR(i, 0, n) {
    /**
     * Functional Utility: Substring detection.
     * Case 1: Detects increment operation regardless of position.
     * Case 2: Detects decrement operation regardless of position.
     */
    if (input.at(i).find("++") != std::string::npos) {
      X++;
    } else if (input.at(i).find("--") != std::string::npos) {
      X--;
    }
  }

  // Output the final state of the virtual register.
  cout << X;

  return 0;
}
