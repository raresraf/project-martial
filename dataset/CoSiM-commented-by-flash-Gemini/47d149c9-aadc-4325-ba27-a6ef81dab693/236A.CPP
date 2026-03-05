/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" problem (Codeforces 236A).
 *
 * This program reads a username (string), counts the number of unique
 * characters in it, and then determines if this count is even or odd.
 * Based on the parity, it prints a specific message.
 *
 * Algorithm: Sorting and counting unique characters. The string is sorted
 * to easily count unique characters by comparing adjacent elements.
 * Time Complexity: O(L log L) due to sorting, where L is the length of the string.
 * Space Complexity: O(L) to store the string.
 */
#include <algorithm>
#include <iostream>

using namespace std;
/**
 * @brief Main function to determine if a username should "CHAT WITH HER!" or "IGNORE HIM!".
 *
 * Reads a username as a string, sorts its characters, counts the number
 * of unique characters, and based on whether this count is even or odd,
 * prints the corresponding message.
 *
 * @return 0 on successful execution.
 */
int main() {

  // Declare a string 's' to store the input username.
  string s;
  // Declare an integer 'a' (unused).
  int a;
  // Read the username into 's'.
  cin >> s;

  // Sort the characters of the string 's' in alphabetical order.
  // This helps in easily counting unique characters as identical characters will be adjacent.
  sort(s.begin(), s.end());
  // Initialize 'count' to 0 to store the number of unique characters.
  int count = 0;
  /**
   * @brief Block Logic: Counts the number of unique characters in the sorted string.
   *
   * Iterates through the sorted string, incrementing 'count' only when a character
   * is different from the next one, effectively counting distinct characters.
   * Pre-condition: 's' is sorted.
   * Invariant: 'count' accurately reflects the number of unique characters encountered so far.
   */
  for (int i = 0; i < s.size(); i++) {
    // If the current character is different from the next one, it's a unique character.
    if (s[i] != s[i + 1]) {
      count++;
    }
  }
  // Pre-condition: 'count' holds the number of unique characters.
  // Block Logic: Determines the message to print based on the parity of 'count'.
  if (count % 2 == 0)
    cout << "CHAT WITH HER!"; // If the count of unique characters is even.
  else
    cout << "IGNORE HIM!"; // If the count of unique characters is odd.
  return 0;
}
