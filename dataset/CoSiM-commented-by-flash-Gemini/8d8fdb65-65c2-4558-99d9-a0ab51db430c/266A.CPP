/**
 * @8d8fdb65-65c2-4558-99d9-a0ab51db430c/266A.CPP
 * @brief This program solves the "Stones on the Table" problem (Codeforces 266A).
 * It counts the number of times a stone on a table has the same color as the stone
 * immediately preceding it. The input is the number of stones and their sequence of colors.
 * Algorithm: Linear scan for consecutive identical characters.
 * Time Complexity: O(N), where N is the number of stones (characters in the input sequence).
 * Space Complexity: O(1) for storing a few variables, excluding input string storage by `cin`.
 */

#include <bits/stdc++.h> // Includes various standard libraries, commonly used in competitive programming.
using namespace std;

int main() {
  // @var s: This variable is declared but not used.
  string s;
  // @var f: Stores the current stone's color (character).
  // @var l: Stores the previous stone's color (character).
  char f, l;
  // @var i, j, m: Loop counters or temporary variables (j, m declared but not used).
  // @var x: Counter for the number of adjacent stones of the same color.
  // @var n: Total number of stones.
  long int i, j, x, m, n;
  
  // Block Logic: Read the number of stones 'n' continuously until EOF.
  // Invariant: 'n' represents the number of stones for the current test case.
  while (cin >> n) {
    x = 0; // Inline: Reset the counter for same-colored adjacent stones for each test case.
    
    // Block Logic: Iterate through the 'n' stones, reading their colors and checking for adjacency.
    // Invariant: 'i' counts from 1 to 'n', 'l' holds the previous character, 'f' holds the current.
    // Pre-condition: 'n' specifies the number of stone colors to read.
    for (i = 1; i <= n; i++) {
      cin >> f; // Read the current stone's color.
      // Block Logic: If it's not the first stone, compare with the previous stone.
      if (i != 1) {
        // Inline: If the current stone has the same color as the previous one, increment counter 'x'.
        if (l == f)
          x++;
      }
      l = f; // Inline: Update 'l' to the current stone's color for the next iteration.
    }
    // Post-condition: Print the total count of adjacent same-colored stones for the current test case.
    cout << x << endl;
  }

  return 0; // Explicitly return 0 for successful execution.
}
