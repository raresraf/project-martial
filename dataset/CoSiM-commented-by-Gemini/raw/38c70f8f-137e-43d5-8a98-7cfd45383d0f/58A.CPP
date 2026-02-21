/**
 * @file 58A.CPP
 * @brief Checks if the word "hello" is a subsequence of a given string.
 *
 * This program reads a string from standard input and determines if the word "hello"
 * can be obtained by deleting zero or more characters from the input string.
 * This is a classic subsequence problem.
 *
 * Algorithm:
 * 1. Define the target string `b` as "hello".
 * 2. Read the source string `a` from the user.
 * 3. Use a pointer `j` to track the current character in the target string `b`.
 * 4. Iterate through the source string `a` with a pointer `i`.
 * 5. If the character `a[i]` matches the character `b[j]`, it means the next
 *    character of the subsequence has been found, so increment `j`.
 * 6. A counter `pas` tracks the number of matched characters. When a character
 *    matches, `pas` is incremented.
 * 7. If `pas` reaches 5, it means all characters of "hello" have been found in
 *    the correct order, and the loop can terminate early.
 * 8. After the loop, if `pas` is 5, print "YES"; otherwise, print "NO".
 *
 * Time Complexity: O(N), where N is the length of the input string `a`, as it
 * involves a single pass through the string.
 * Space Complexity: O(N) to store the input string `a`.
 */
#include <iostream>
using namespace std;

string a, b = "hello";
int j = 0, pas = 0;

int main() {
  // Read the source string.
  cin >> a;
  
  /**
   * @brief Iterates through the source string to find the "hello" subsequence.
   * Pre-condition: `a` is the source string, `b` is the target "hello".
   * Invariant: `j` points to the next character to be found in `b`. `pas` counts
   * how many characters have been found so far.
   */
  for (int i = 0; i < a.size(); i++) {
    // If a character in the source string matches the current target character...
    if (a[i] == b[j]) {
      // ...move to the next character in the target string.
      j++;
      // Increment the count of matched characters.
      pas++;

      // Optimization: if all characters are found, no need to continue.
      if (pas == 5) {
        break;
      }
    }
  }
  
  // Check if the entire "hello" subsequence was found.
  if (pas == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
