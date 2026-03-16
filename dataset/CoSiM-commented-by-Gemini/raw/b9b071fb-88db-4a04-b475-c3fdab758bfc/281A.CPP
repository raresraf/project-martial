/**
 * @file 281A.CPP
 * @brief Solution to the "Word Capitalization" problem (Codeforces 281A).
 *
 * @details
 * This program reads a single word and capitalizes its first letter. If the word
 * is empty, it remains unchanged. The rest of the word is not modified.
 *
 * The code is well-structured, with the core logic isolated in a `solve` function
 * and the output handled by an `answer` helper function. The `solve` function
 * modifies a copy of the input string.
 *
 * Algorithm:
 * 1. Read a word from standard input.
 * 2. Pass the word to a function that checks if it's non-empty.
 * 3. If non-empty, convert the character at the first index (index 0) to uppercase.
 * 4. Print the modified word.
 *
 * Time Complexity: O(L), where L is the length of the word. This is because the
 *                  string is passed by value to `solve`, requiring a copy of
 *                  length L to be made.
 * Space Complexity: O(L), to store the original string and its copy.
 */
#include <iostream>
#include <string>
#include <cctype> // For toupper

/**
 * @brief Prints the given string to standard output, followed by a newline.
 * @param v The string to be printed.
 */
void answer(const std::string &v) { std::cout << v << '\n'; }

/**
 * @brief Capitalizes the first letter of a given word.
 * @param word The input string. A copy of the string is passed.
 */
void solve(std::string word) {
  // Pre-condition: Ensure the string is not empty before accessing its first character.
  if (!word.empty())
    word[0] = toupper(word[0]);

  answer(word);
}

/**
 * @brief Main function: handles input and initiates the solving process.
 */
int main() {
  std::string word;
  std::cin >> word;

  solve(word);

  return 0;
}