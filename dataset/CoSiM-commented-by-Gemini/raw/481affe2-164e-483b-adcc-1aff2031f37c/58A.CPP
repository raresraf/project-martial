/**
 * @file 58A.CPP
 * @brief Solution for a subsequence problem ("Chat room").
 * @details The program checks if the word "hello" can be formed by deleting zero or
 * more characters from a given input string. In other words, it checks if "hello"
 * is a subsequence of the input string.
 *
 * Algorithm:
 * The intended algorithm uses a greedy approach with two pointers.
 * 1. One pointer (`i`) iterates through the input string.
 * 2. Another pointer (`j`) tracks the current character to be found in the target
 *    word "hello".
 * 3. As the code iterates through the input string, if a character matches the
 *    target character `hello[j]`, the `j` pointer is advanced.
 * 4. This process continues until either the end of the input string is reached or
 *    all characters of "hello" have been found in order.
 * 5. If all 5 characters of "hello" are found, the result is "YES". Otherwise, it's "NO".
 *
 * Implementation Bug:
 * The final check `if (pas == 6)` is incorrect. The counter `pas` tracks the
 * number of matched characters and can reach a maximum of 5 (the length of "hello").
 * The condition should be `if (pas == 5)`. Due to this bug, the program will
 * incorrectly output "NO" for all valid inputs.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
using namespace std;

string a, b = "hello";
int j = 0, pas = 0;
int main() {
  cin >> a;
  
  /**
   * @block
   * @description This loop greedily searches for the characters of "hello" in the
   * input string `a` in the correct order.
   */
  for (int i = 0; i < a.size(); i++) {
    // Check if the current character in `a` matches the target character in `b`.
    if (a[i] == b[j]) {
      // If a match is found, advance to the next character in the target string "hello".
      j++;
      pas++;

      // Optimization: if all 5 characters are found, we can exit early.
      if (pas == 5) {
        break;
      }
    }
  }
  
  /**
   * @block
   * @description This condition is intended to check if the subsequence was found.
   * BUG: It incorrectly checks if `pas == 6`. Since "hello" has 5 letters,
   * `pas` will be 5 on success. This condition will always be false.
   */
  if (pas == 6) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}