/**
 * @file 58A.CPP
 * @brief Checks if the string "hello" is a subsequence of the input string.
 *
 * This program reads a string from standard input and determines if the
 * characters of "hello" appear in the input string in the correct order,
 * but not necessarily consecutively.
 */
#include <iostream>
#include <string>

/**
 * @brief Prints "YES" if the condition is true, "NO" otherwise.
 * @param v The boolean condition.
 */
void answer(bool v) {
  constexpr const char *s[2] = {"NO", "YES"};
  std::cout << s[v] << '
';
}

/**
 * @brief Solves the "hello" subsequence problem.
 * @param input The string to check.
 *
 * This function iterates through the input string and checks if it contains
 * "hello" as a subsequence. It uses a pointer `i` to keep track of the
 * current character in "hello" that it is looking for.
 */
void solve(const std::string &input) {
  constexpr char m[] = "hello";

  size_t i = 0; // Pointer to the current character in "hello"
  for (const char c : input) {
    // If the current character in the input matches the character we are looking for
    if (c == m[i] && ++i == 5)
      return answer(true); // If we have found all characters of "hello"
  }

  answer(false); // If we reached the end of the input without finding "hello"
}

int main() {
  std::string input;
  std::cin >> input;

  solve(input);

  return 0;
}
