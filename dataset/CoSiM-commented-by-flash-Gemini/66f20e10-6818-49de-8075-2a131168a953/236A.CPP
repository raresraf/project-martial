/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A: Boy or Girl.
 * @details This program classifies a given username as "boy" or "girl" based on the parity
 * of the number of distinct characters it contains. If the count of distinct characters
 * is even, it's a "girl" username ("CHAT WITH HER!"). If the count is odd, it's a "boy"
 * username ("IGNORE HIM!"). The program uses a frequency array to count distinct characters.
 * @algorithm Character frequency counting and parity check.
 * Time Complexity: $O(L + \alpha)$ where L is the length of the username string and $\alpha$ is the size of the alphabet (26 for lowercase English letters).
 * Space Complexity: $O(\alpha)$ for the frequency array.
 */
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
using namespace std;
/**
 * @file 236A.CPP
 * @brief Solution for Codeforces Problem 236A: Boy or Girl.
 * @details This program classifies a given username as "boy" or "girl" based on the parity
 * of the number of distinct characters it contains. If the count of distinct characters
 * is even, it's a "girl" username ("CHAT WITH HER!"). If the count is odd, it's a "boy"
 * username ("IGNORE HIM!"). The program uses a frequency array to count distinct characters.
 * @algorithm Character frequency counting and parity check.
 * Time Complexity: $O(L + \alpha)$ where L is the length of the username string and $\alpha$ is the size of the alphabet (26 for lowercase English letters).
 * Space Complexity: $O(\alpha)$ for the frequency array.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <math.h>   // Included but not used in this specific solution.
#include <stdio.h>  // Required for scanf.
#include <string.h> // Required for strlen.
using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the username classification logic.
   */
  char a[100]; // @var a: Character array to store the input username string. Max size 99 characters + null terminator.
  int b[26];   // @var b: Frequency array to count occurrences of each lowercase English letter. Index 0 for 'a', 25 for 'z'.
  int flag = 0; // @var flag: Declared but not used in the current implementation.
  int i, j,    // @var i, j: Loop counters and temporary variables for character conversion.
      c = 0;   // @var c: Counter for the number of distinct characters in the username.

  /**
   * @brief Initializes the frequency array `b` to all zeros.
   * @block_logic Iterates through the `b` array and sets each element to 0.
   * @invariant Each `b[i]` is 0 before processing the username.
   */
  for (i = 0; i < 26; ++i) {
    b[i] = 0;
  }

  scanf("%s", a); // Reads the username string into character array `a`.

  /**
   * @brief Populates the frequency array `b` based on characters in the username `a`.
   * @block_logic Iterates through each character of the username, converts it to a 0-25 index, and increments its count.
   * @invariant `b[index]` stores the count of the character corresponding to `index`.
   */
  for (i = 0; i < strlen(a); i++) {
    j = a[i];    // Get ASCII value of character.
    j = j - 97;  // Convert ASCII lowercase letter to 0-25 index ('a' -> 0).
    b[j]++;      // Increment frequency count for this character.
  }

  /**
   * @brief Counts the number of distinct characters in the username.
   * @block_logic Iterates through the frequency array `b`.
   * @invariant `c` accumulates the count of elements in `b` that are greater than 0.
   */
  for (i = 0; i < 26; i++) {
    if (b[i] > 0) // If a character's frequency is greater than 0, it means it's present.
      c++;        // Increment the distinct character counter.
  }

  /**
   * @brief Classifies the username based on the parity of distinct character count.
   * @block_logic Checks if `c` (distinct character count) is even or odd.
   * @invariant Prints "CHAT WITH HER!" if `c` is even, otherwise "IGNORE HIM!".
   */
  if (c % 2 == 0)
    cout << "CHAT WITH HER!" << endl; // Output for even number of distinct characters.
  else
    cout << "IGNORE HIM!" << endl;  // Output for odd number of distinct characters.

  return 0; // Indicates successful execution.
}