/**
 * @file 96A.CPP
 * @brief Solution to the "Football" competitive programming problem (Codeforces 96A).
 * This program determines if a team is in a "dangerous" situation. A dangerous situation
 * is defined as having 7 or more consecutive players of the same team (represented by
 * identical characters, e.g., '0000000' or '1111111') in a sequence.
 * Algorithm: Iterative string traversal with consecutive character counting.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries (e.g., iostream, string, algorithm).

using namespace std;
#define EPS (1e-7) // Epsilon for floating-point comparisons (unused in this problem).
#define INF (1e9)  // Represents infinity, typically for graph problems (unused).
#define PI (acos(-1)) // Value of Pi (unused).

typedef long long ll; // Defines 'll' as a shorthand for 'long long'.
#define endl '\n'    // Redefines `endl` to be just a newline character for potentially faster output.

// Arrays for 4-directional movement in grids (unused in this problem).
int dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};

const int MOD = 1000000007; // Modulo constant, typically for operations in modular arithmetic (unused).

//////////////////////////////////////////

string s; // Global string 's' to store the input sequence.

/**
 * @brief Reads the input string into the global variable `s`.
 */
void Input() { cin >> s; }

/**
 * @brief Implements the core logic to check for a dangerous situation.
 * It iterates through the string `s` and counts consecutive identical characters.
 */
void Solve() {

  char c = s[0]; // Stores the current character being checked for consecutive occurrences.
  int cnt = 1;   // Counts consecutive identical characters.
  int len = s.size(); // Stores the length of the input string for loop bounds.
  
  /**
   * Block Logic: Iterates through the string `s` starting from the second character.
   * Invariant: `c` holds the character being counted, `cnt` is its consecutive count,
   * and `len` represents the total length for iteration.
   */
  for (int i = 1; i < len; i++) {

    // Conditional Logic: Early exit if a dangerous situation (7 or more consecutive characters) is already found.
    if (cnt >= 7) {
      break;
    }
    // Conditional Logic: Checks if the current character `s[i]` is the same as the character `c`.
    if (c == s[i]) {
      cnt++; // Increments the consecutive count.

    } else {
      // If characters are different, reset the count for the new character.
      c = s[i]; // Updates the character being counted.
      cnt = 1;  // Resets the consecutive count to 1.
    }
  }

  // Final Conditional Logic: Prints "YES" if a dangerous situation was found, "NO" otherwise.
  if (cnt >= 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}

int main() {
  // Functional Utility: Optimizes C++ standard stream I/O operations.
  // `cin.tie(0)` prevents `cin` from flushing `cout` before each input.
  // `ios::sync_with_stdio(false)` unties C++ streams from C stdio.
  cin.tie(0);
  ios::sync_with_stdio(false);
  
  Input(); // Calls the function to read input.
  Solve(); // Calls the function to solve the problem.

  return 0; // Indicates successful program execution.
}
