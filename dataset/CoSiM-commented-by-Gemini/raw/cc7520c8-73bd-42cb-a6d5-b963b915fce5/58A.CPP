/**
 * @file 58A.CPP
 * @brief Solution for the "Chat room" problem.
 * @details This program checks if the word "hello" can be spelled out by taking
 * characters from an input string in order, but not necessarily contiguously.
 * This is a classic subsequence problem.
 *
 * Algorithm:
 * The program uses a greedy approach to find if "hello" is a subsequence of the
 * input string. It maintains a pointer to the current character it is looking for
 * in the target word "hello". It iterates through the input string, and if the
 * current character matches the target character, it advances the target pointer.
 * If the target pointer traverses the entire word "hello", the subsequence exists.
 *
 * Time Complexity: O(N), where N is the length of the input string, because we
 * iterate through the input string only once.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>
#include <string>

// A common macro in competitive programming for cleaner output statements.
#define LOG(x) cout << x << "
"

using namespace std;

// Forward declaration of the parsing function.
bool parseHello(string);

int main() {
  string str;
  // Read the input string from the user.
  cin >> str;
  // Call the function and print YES/NO based on the result.
  parseHello(str) ? LOG("YES") : LOG("NO");
  return 0;
}

/**
 * @brief Checks if "hello" is a subsequence of the given string s.
 * @param s The input string to search within.
 * @return True if "hello" is a subsequence, false otherwise.
 */
bool parseHello(string s) {
  // The target subsequence we are looking for.
  char hello[] = {'h', 'e', 'l', 'l', 'o'};
  // 'i' is a state pointer, tracking our progress in matching the 'hello' array.
  int i = 0;

  /**
   * Block Logic: Iterate through the input string to find the 'hello' subsequence.
   * This is a greedy algorithm.
   */
  for (char c : s) {
    // Pre-condition: 'i' points to the next character we need to find in 'hello'.
    // If the current character 'c' from the input string matches our target...
    if (i < 5 && c == hello[i]) {
      // ...we've found it, so we advance our target pointer.
      i++;
    }
    // Post-condition: 'i' has been advanced if a match was found.

    // Optimization: If we have found all characters, we can exit early.
    if (i == 5) {
      return true;
    }
  }

  // If the loop completes and we haven't found all 5 characters, it's not a subsequence.
  return false;
}
