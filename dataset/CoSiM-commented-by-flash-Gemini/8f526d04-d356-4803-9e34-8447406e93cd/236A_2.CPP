/**
 * @file 236A_2.CPP
 * @brief Solution for Codeforces Problem 236A: Boy or Girl (Vector and Unique Implementation).
 * @details This program classifies a given username as "boy" or "girl" based on the parity
 * of the number of distinct characters it contains. If the count of distinct characters
 * is even, it's a "girl" username ("CHAT WITH HER!"). If the count is odd, it's a "boy"
 * username ("IGNORE HIM!"). This implementation converts the input string to a character
 * vector, sorts it, and then uses `std::unique` to find the number of distinct characters.
 * @algorithm Sorting and `std::unique` for distinct character counting.
 * Time Complexity: $O(L \log L)$ due to sorting, where L is the length of the username string.
 * Space Complexity: $O(L)$ to store the character vector.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 236A_2.CPP
 * @brief Solution for Codeforces Problem 236A: Boy or Girl (Vector and Unique Implementation).
 * @details This program classifies a given username as "boy" or "girl" based on the parity
 * of the number of distinct characters it contains. If the count of distinct characters
 * is even, it's a "girl" username ("CHAT WITH HER!"). If the count is odd, it's a "boy"
 * username ("IGNORE HIM!"). This implementation converts the input string to a character
 * vector, sorts it, and then uses `std::unique` to find the number of distinct characters.
 * @algorithm Sorting and `std::unique` for distinct character counting.
 * Time Complexity: $O(L \log L)$ due to sorting, where L is the length of the username string.
 * Space Complexity: $O(L)$ to store the character vector.
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers (e.g., iostream, string, vector, algorithm).
using namespace std;     // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: orchestrates the username classification logic.
   */
  string s; // @var s: The input username string.
  cin >> s; // Reads the input username string.

  vector<char> v; // @var v: A dynamic array (vector) to store individual characters of the username.

  /**
   * @brief Populates the vector `v` with characters from the input string `s`.
   * @block_logic Iterates through each character of `s` and adds it to `v`.
   * @invariant `v` contains all characters of `s` in their original order.
   */
  for (int i = 0; i < s.size(); i++) {
    v.push_back(s[i]); // Adds the current character to the vector.
  }

  /*
   * The commented out block below shows an alternative way to initialize the vector
   * from a C-style string, or a direct character array initialization.
   * OR,  char s[100];
   *         cin >> s;
   *         len = strlen(s);
   *         vector<char>v(s,s+len);
   */

  /**
   * @brief Sorts the vector `v` to group identical characters together.
   * @details Sorting is a prerequisite for `std::unique` to work correctly.
   * @param v.begin() Iterator to the beginning of the vector.
   * @param v.end()   Iterator to the end of the vector.
   */
  sort(v.begin(), v.end());

  vector<char>::iterator it; // @var it: An iterator to store the new end of the unique range.

  /**
   * @brief Removes duplicate characters from the sorted vector `v` and returns an iterator to the new end.
   * @details `std::unique` rearranges the elements in `v` such that all unique elements are at the beginning
   *          of the range, and returns an iterator to the element that follows the last unique element.
   *          It does not actually resize the container.
   * @param v.begin() Iterator to the beginning of the range to process.
   * @param v.end()   Iterator to the end of the range to process.
   * @return An iterator to the element that would follow the last element of the new range of unique elements.
   */
  it = unique(v.begin(), v.end());

  /**
   * @brief Calculates the number of distinct characters.
   * @details `std::distance` calculates the number of elements between `v.begin()` and `it`,
   *          which represents the count of unique characters.
   */
  int curent_length = distance(v.begin(), it); // @var curent_length: Stores the count of distinct characters.

  /**
   * @brief Classifies the username based on the parity of the distinct character count.
   * @block_logic Checks if `curent_length` is even or odd.
   * @invariant Prints "CHAT WITH HER!" if `curent_length` is even, otherwise "IGNORE HIM!".
   */
  if (curent_length % 2 == 0) {
    cout << "CHAT WITH HER!"; // Output for even number of distinct characters.
  } else {
    cout << "IGNORE HIM!";  // Output for odd number of distinct characters.
  }

  // Functional Utility: Debugging output for the distinct character count.
  cout << "\n" << distance(v.begin(), it) << endl; // Prints the count of distinct characters followed by a newline.

  // Functional Utility: Debugging output for the (potentially modified) sorted vector elements.
  // Note: This loop prints characters from the vector `v` which might still contain duplicate elements
  //       after the unique elements, as `std::unique` only shuffles them to the end, it doesn't erase them.
  for (int i = 0; i < s.size(); i++) {
    cout << v[i] << " ";
  }

  return 0; // Indicates successful execution.
}
