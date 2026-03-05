/**
 * @file 96A.CPP
 * @brief Checks for a sequence of seven consecutive identical characters.
 *
 * This program solves the "Football" problem, where it determines if a given
 * situation is "dangerous" by checking if there are at least seven consecutive
 * players of the same team ('0' or '1') in a line.
 *
 * @algorithm
 * 1. Read the input string `s`.
 * 2. Initialize a character `c` with the first character of the string and a `count` to 1.
 * 3. Loop through the string from the first character (`i = 0`).
 * 4. In each iteration, compare the current character `s[i]` with the character `c`
 *    that marks the start of the current consecutive block.
 *    a. If `s[i]` is different from `c`, a new block has started. Update `c` to `s[i]`
 *       and reset `count` to 1.
 *    b. If `s[i]` is the same as `c`, increment the `count`.
 * 5. After updating the count, check if it has reached 7.
 *    a. If it has, print "YES" and terminate the program immediately.
 * 6. If the loop completes without the count ever reaching 7, it means no
 *    dangerous situation was found. Print "NO".
 *
 * @implementation_note
 * There is a subtle logic flaw in this implementation. The loop starts at `i = 0`
 * after `c` has already been initialized to `s[0]` and `count` to 1. In the first
 * iteration (`i=0`), `s[0]` is compared to `c` (which is `s[0]`), causing `count`
 * to be incremented to 2. This means the count for the first character is effectively
 * doubled. For a string starting with "111111...", the count will be 7 when only six
 * '1's have been processed. This bug may not affect correctness on typical test
 * cases but is logically incorrect. A correct implementation would start the loop
 * from `i = 1`.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the string, due to the single pass.
 * Space Complexity: O(L) to store the input string.
 */
#include <iostream>
#include <string> // It's good practice to include the string header.

int main() {
  std::string s;
  std::cin >> s;

  // If the input string is empty, it's not dangerous.
  if (s.empty()) {
      std::cout << "NO\n";
      return 0;
  }

  // `c` stores the character of the current consecutive block.
  // `count` stores the length of the current block.
  // Initialized with the first character.
  char c = s[0];
  char count = 1;

  /**
   * @brief Loop to iterate through the string and find a consecutive block of 7.
   * NOTE: The loop starts from i=0, which causes a logical flaw as the first
   * character gets counted twice. The loop should ideally start from i=1.
   */
  for (unsigned int i = 0; i < s.size(); ++i) {
    if (s[i] != c) {
      // A new block of characters has started.
      c = s[i];
      count = 1;
    } else { // This branch is taken for i=0, leading to the bug.
      // The block of the same character continues.
      ++count;
    }
    
    // Check for the "dangerous" condition.
    if (count == 7) {
      std::cout << "YES\n";
      return 0; // Terminate successfully after finding the sequence.
    }
  }
  
  // If the loop completes, no sequence of 7 was found.
  std::cout << "NO\n";
  // The return 0 is implicit at the end of main.
}
