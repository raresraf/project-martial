/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem from Codeforces.
 * @details The program determines if the word "hello" exists as a subsequence
 * in a given input string. A subsequence maintains the relative order of
 * characters but does not require them to be contiguous.
 *
 * @algorithm
 * A two-pointer technique is used to find the subsequence.
 * 1. Define the target subsequence, "hello".
 * 2. Initialize a pointer `a` for the "hello" string and a pointer `b` for the
 *    input string. Both start at 0.
 * 3. Iterate through the input string using pointer `b`.
 * 4. In each iteration, check if the character at the input string's current
 *    position (`line[b]`) matches the character at the "hello" string's
 *    current position (`greeting[a]`).
 * 5. If they match, it means we have found the next character of our
 *    subsequence, so we advance the pointer `a`.
 * 6. The pointer `b` is always advanced to move to the next character in the
 *    input string.
 * 7. If pointer `a` reaches the end of "hello", the complete subsequence has
 *    been found. Set the result to "YES" and terminate.
 * 8. If the loop finishes before `a` reaches the end, the subsequence was not
 *    found, and the result remains "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, as we
 * make a single pass through it.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
using namespace std;

int main() {
  const string greeting = "hello"; // The target subsequence.
  string line;                     // The input string to search within.
  getline(cin, line);
  
  // 'a' is the pointer for the `greeting` string.
  // 'b' is the pointer for the `line` string.
  int a = 0, b = 0;
  
  string output = "NO"; // Default result.
  
  /**
   * @block
   * @brief Iterate through the input string to find the subsequence.
   */
  while (b < line.size()) {
    // Compare current characters and advance the input string pointer 'b'.
    if (line[b++] == greeting[a]) {
      // If a match is found, advance the subsequence pointer 'a'.
      a++;
    }
    
    // Check if the entire subsequence has been found.
    if (a >= greeting.size()) {
      output = "YES";
      break;
    }
  }
  
  cout << output << endl;
  return 0;
}
