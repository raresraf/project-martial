/**
 * @file 236A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Boy or Girl").
 * This program determines a user's gender based on a username. If the number
 * of distinct characters in the username is even, the user is identified as
 * female ("CHAT WITH HER!"). If it's odd, the user is male ("IGNORE HIM!").
 *
 * @algorithm The algorithm counts the number of unique characters in the input
 * string using a map and then checks the parity (even or odd) of this count.
 *
 * @time_complexity O(L * log K), where L is the length of the string and K is
 * the number of distinct characters. The log factor comes from map insertions.
 * @space_complexity O(K) to store the distinct characters in the map.
 */
#include <iostream>
#include <map>
#include <string>

using namespace std;

// Function prototypes
int get_distinct_char_count(const string &);
bool is_female(const string &);

/**
 * @brief Main function to read input and print the result.
 */
int main(void) {
  string username;
  cin >> username;

  if (is_female(username))
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}

/**
 * @brief Counts the number of distinct characters in a string.
 * @param str The input string.
 * @return The total count of unique characters in the string.
 *
 * This function uses a std::map to keep track of characters it has already seen.
 * The size of the map at the end represents the number of distinct characters.
 */
int get_distinct_char_count(const string &str) {
  map<char, int> char_frequency;

  for (char ch : str)
    // The `count` method checks for key existence.
    if (!char_frequency.count(ch))
      // If the character is not in the map, insert it.
      char_frequency.insert(make_pair(ch, 1));
    else
      // This else block is redundant for just counting distinct characters
      // but is part of the original code.
      char_frequency[ch]++;

  return char_frequency.size();
}

/**
 * @brief Determines if the user is female based on username criteria.
 * @param username The user's name string.
 * @return `true` if the count of distinct characters is even, `false` otherwise.
 */
bool is_female(const string &username) {
  if (get_distinct_char_count(username) % 2 == 0)
    return true;
  else
    return false;
}