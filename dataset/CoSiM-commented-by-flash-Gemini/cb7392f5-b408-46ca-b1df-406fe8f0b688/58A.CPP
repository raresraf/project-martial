/**
 * @file 58A.CPP
 * @brief This program determines if the word "hello" can be formed as a subsequence
 * from a given input string. A subsequence means the characters appear in the correct
 * order, but not necessarily consecutively.
 * Algorithm: State-machine based character matching. It uses flags to track the sequential
 * discovery of each character ('h', 'e', 'l', 'l', 'o') within the input string.
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass.
 * Space Complexity: O(N) for storing the input string, plus O(1) for flags.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.
#include <stdlib.h>      // C standard library header, not explicitly used for this logic.
#include <string>        // Required for std::string.

using namespace std;

int main() {
  string s; // Stores the input string.

  // Flags to track the discovery of each character of "hello" in sequence.
  // Initialized to 0 (not found). They are incremented to 1 when the character is found.
  int h = 0, e = 0, l = 0, l1 = 0, o = 0;

  // Reads the input string from standard input.
  cin >> s;

  /**
   * Block Logic: Iterates through the input string 's' to find the characters of "hello" in order.
   * It acts like a state machine, progressing through 'h', 'e', 'l', 'l', 'o'.
   * Each 'else if' condition ensures that a character is only matched if the previous
   * characters in the sequence have already been found.
   * Invariant: Flags (`h`, `e`, `l`, `l1`, `o`) track the furthest character of "hello" found as a subsequence so far.
   */
  for (int i = 0; i < s.size(); i++) {
    // If 'h' is the current character and hasn't been found yet.
    if (s[i] == 'h' && h == 0) {
      h++; // Mark 'h' as found.
    }
    // Else if 'e' is the current character and 'h' has been found but 'e' hasn't.
    else if (s[i] == 'e' && h == 1 && e == 0) {
      e++; // Mark 'e' as found.
    }
    // Else if 'l' is the current character and 'e' has been found but the first 'l' hasn't.
    else if (s[i] == 'l' && e == 1 && l == 0) {
      l++; // Mark the first 'l' as found.
    }
    // Else if 'l' is the current character and the first 'l' has been found but the second 'l' hasn't.
    else if (s[i] == 'l' && l == 1 && l1 == 0) {
      l1++; // Mark the second 'l' as found.
    }
    // Else if 'o' is the current character and the second 'l' has been found but 'o' hasn't.
    else if (s[i] == 'o' && l1 == 1 && o == 0) {
      o++; // Mark 'o' as found.
    }
  }

  /**
   * Conditional Logic: After checking all characters in the string, determines if "hello" was formed.
   * If the sum of all flags is 5 (meaning each character 'h', 'e', 'l', 'l', 'o' was found in sequence),
   * then "YES" is printed. Otherwise, "NO" is printed.
   */
  if (h + e + l + l1 + o == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  // Standard program exit.
  return 0;
}
