/**
 * @file 236A.CPP
 * @brief This program classifies a user as "female" or "male" based on the parity
 * of the count of distinct characters in their username. If the number of distinct
 * characters is even, the user is classified as female; otherwise, male.
 *
 * Algorithm: Character frequency counting and parity check.
 * 1. Read the username as a string.
 * 2. Count the number of unique characters present in the username.
 * 3. If the count of unique characters is even, print "CHAT WITH HER!".
 * 4. If the count is odd, print "IGNORE HIM!".
 * Time Complexity: O(L) where L is the length of the username string, as it involves a single pass to count character frequencies.
 * Space Complexity: O(C) where C is the size of the character set (e.g., 26 for lowercase English letters, 256 for ASCII), to store character frequencies in a map.
 */

#include <iostream>
#include <map>
#include <string>

using namespace std;

// Function declarations for modularity and readability.
int get_distinct_char_count(const string &);
bool is_female(const string &);

/**
 * @brief Main function of the program.
 * Reads the username, determines the gender classification, and prints the result.
 * @pre The input `username` is a string.
 * @post Prints "CHAT WITH HER!" if the username is classified as female, "IGNORE HIM!" otherwise.
 * @return 0 upon successful execution.
 */
int main(void) {
  string username;
  // Functional Utility: Reads the username from standard input.
  cin >> username;

  // Block Logic: Determines the output message based on the gender classification.
  // Pre-condition: 'username' has been read and processed by the 'is_female' function.
  if (is_female(username))
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}

/**
 * @brief Counts the number of distinct characters in a given string.
 * Uses a map to store character frequencies.
 * @param str The input string.
 * @pre `str` is a valid string.
 * @post Returns the total count of unique characters in `str`.
 * @return An integer representing the number of distinct characters.
 */
int get_distinct_char_count(const string &str) {
  // Functional Utility: A map to store the frequency of each character. Keys are characters, values are their counts.
  map<char, int> char_frequency;

  // Block Logic: Iterates through each character of the string to build a frequency map.
  // Invariant: `char_frequency` accurately stores the counts of distinct characters encountered so far.
  for (char ch : str)
    // Block Logic: Checks if the character is encountered for the first time.
    // Pre-condition: `ch` is the current character from the string.
    if (!char_frequency.count(ch))
      char_frequency.insert(make_pair(ch, 1)); // Functional Utility: Inserts the character with a count of 1 if it's new.
    else
      char_frequency[ch]++; // Functional Utility: Increments the count for an existing character.

  // Functional Utility: The size of the map directly corresponds to the number of distinct characters.
  return char_frequency.size();
}

/**
 * @brief Determines if a user is "female" based on the distinct character count in their username.
 * A user is "female" if the count of distinct characters is even.
 * @param username The username string.
 * @pre `username` is a valid string.
 * @post Returns true if the distinct character count is even, false otherwise.
 * @return A boolean value indicating the gender classification.
 */
bool is_female(const string &username) {
  // Block Logic: Checks the parity of the distinct character count.
  // Pre-condition: `get_distinct_char_count(username)` returns the number of unique characters.
  if (get_distinct_char_count(username) % 2 == 0)
    return true;
  else
    return false;
}
