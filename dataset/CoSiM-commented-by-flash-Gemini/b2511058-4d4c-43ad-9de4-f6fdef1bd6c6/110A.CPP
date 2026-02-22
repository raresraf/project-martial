/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A: Nearly Lucky Number (Alternative Implementation).
 * @details This program determines if a given integer 'n' is a "nearly lucky number".
 * A number is considered nearly lucky if, after counting the occurrences of digits '4' and '7' within 'n',
 * this resulting count itself consists *only* of digits '4' and '7'. The program first counts
 * the lucky digits in the input number and then checks if the decimal representation of this count
 * contains only '4's and '7's.
 * @algorithm Digit counting and string-based digit validation.
 * Time Complexity: $O(L_n + L_c)$ where $L_n$ is the number of digits in the input number and $L_c$ is the number of digits in the count of lucky digits.
 * Space Complexity: $O(L_n + L_c)$ to store the input number as a string and the count as a string.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A: Nearly Lucky Number (Alternative Implementation).
 * @details This program determines if a given integer 'n' is a "nearly lucky number".
 * A number is considered nearly lucky if, after counting the occurrences of digits '4' and '7' within 'n',
 * this resulting count itself consists *only* of digits '4' and '7'. The program first counts
 * the lucky digits in the input number and then checks if the decimal representation of this count
 * contains only '4's and '7's.
 * @algorithm Digit counting and string-based digit validation.
 * Time Complexity: $O(L_n + L_c)$ where $L_n$ is the number of digits in the input number and $L_c$ is the number of digits in the count of lucky digits.
 * Space Complexity: $O(L_n + L_c)$ to store the input number as a string and the count as a string.
 */
#include <bits/stdc++.h> // Standard C++ library includes (e.g., iostream, string, algorithm).

using namespace std; // Using the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function: orchestrates the "nearly lucky number" determination.
   */
  string n; // @var n: The input number as a string to facilitate digit access.
  cin >> n; // Reads the input number as a string.

  int lucky = 0; // @var lucky: Counter for the occurrences of lucky digits ('4' or '7') in the input number.

  /**
   * @brief Counts the occurrences of lucky digits ('4' or '7') in the input number `n`.
   * @block_logic Iterates through each character (digit) of the input string `n`.
   * @invariant `lucky` accumulates the count of '4's and '7's.
   */
  for (int i = 0; i < n.length(); i++) {
    if (n[i] == '4' || n[i] == '7') { // Checks if the current character is '4' or '7'.
      lucky++;                       // Increments the lucky digit counter.
    }
  }

  // Functional Utility: Converts the count of lucky digits into its string representation.
  string luckstr = to_string(lucky); // @var luckstr: The string representation of the `lucky` count.

  /**
   * @brief Validates if the `luckstr` (string representation of the lucky digit count)
   *        consists solely of '4's and '7's.
   * @block_logic Iterates through each character (digit) of `luckstr`.
   * @invariant If any character in `luckstr` is not '4' or '7', it prints "NO" and terminates.
   */
  for (int i = 0; i < luckstr.length(); i++) {
    if (!(luckstr[i] == '4' || luckstr[i] == '7')) { // Checks if the current character is NOT '4' or '7'.
      cout << "NO" << endl;                      // Output "NO" if a non-lucky digit is found in the count.
      return 0;                                  // Terminates the program.
    }
  }

  // Functional Utility: If all digits in `luckstr` are '4' or '7', prints "YES".
  cout << "YES" << endl;
  return 0; // Indicates successful execution.
}
