/**
 * @file 58A.CPP
 * @brief Logic for subsequence detection in a noisy character sequence.
 * 
 * Logic: Determines if the target sequence "hello" exists as a subsequence within 
 * an input string, meaning the characters 'h', 'e', 'l', 'l', 'o' appear in that 
 * specific order, possibly with intermediate noise characters.
 * 
 * Algorithm: Two-pointer greedy subsequence matching.
 * Time Complexity: $O(N)$ where N is the length of the input string.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <iostream>
#include <string>

int main() {
  std::string inputStr, hello, outputStr;
  unsigned int i, j;
  
  // The target pattern to be identified within the source stream.
  hello = "hello";
  // Default classification state (Failure).
  outputStr = "NO";
  
  std::cin >> inputStr;

  /**
   * Block Logic: Single-pass scan of the input string to match the target pattern.
   * Invariant: 'j' represents the current progress index in the "hello" sequence.
   */
  for (i = j = 0; i < inputStr.length(); i++) {
    // Logic: If the current character matches the expected character in the pattern.
    if (inputStr.at(i) == hello.at(j)) {
      /**
       * Functional Utility: Increments pattern pointer and checks for early termination 
       * once all 5 characters are successfully identified in order.
       */
      if (++j == 5) {
        outputStr = "YES";
        break;
      }
    }
  }

  // Output the final classification based on the success of the pattern match.
  std::cout << outputStr << '\n';
  
  return 0;
}
