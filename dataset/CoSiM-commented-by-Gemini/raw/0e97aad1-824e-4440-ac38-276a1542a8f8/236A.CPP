/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem (Codeforces 236A).
 * @author competitive-programmer
 *
 * @details This program determines the gender of a user based on the number
 * of distinct characters in their username. If the number of distinct
 * characters is even, the user is considered female ("CHAT WITH HER!").
 * If it's odd, the user is considered male ("IGNORE HIM!").
 *
 * Algorithm:
 * 1. Read the username as a string.
 * 2. Iterate through the string and count the number of unique characters.
 *    A helper function `is_contain` is used to check for character uniqueness.
 * 3. Check if the total count of unique characters is even or odd.
 * 4. Print the corresponding message.
 *
 * Time Complexity: O(N * M), where N is the length of the username and M is
 * the number of distinct characters found so far. This is because for each
 * character in the input string, `is_contain` performs a linear scan.
 * A more optimal approach using `std::set` or `std::unordered_set` would
 * achieve O(N log M) or O(N) on average.
 * Space Complexity: O(M), where M is the number of distinct characters, to
 * store the unique letters.
 */
#include <iostream>
#include <string>
#include <vector>
#include <set> // Using a set would be more efficient

using namespace std;

/**
 * @brief Checks if a character is present in a vector of characters.
 * @param distinct_letters The vector of characters to search in.
 * @param check The character to search for.
 * @return 1 if the character is found, 0 otherwise.
 * @note This is an O(N) operation. Using a `std::set` for `distinct_letters`
 *       would make this check O(log N).
 */
short is_contain(vector<char> distinct_letters, char check) {

  for (short i = 0; i < distinct_letters.size(); i++) {
    if (distinct_letters[i] == check) {
      return 1;
    }
  }

  return 0;
}

int main() {
  string user_name;
  cin >> user_name;

  // A vector to store the unique characters found in the username.
  vector<char> distinct_letters{};
  
  /**
   * Block Logic: Iterate through the username to find all distinct characters.
   * Logic: For each character in the username, it checks if the character has
   * already been added to the `distinct_letters` vector. If not, it adds it.
   */
  for (short i = 0; i < user_name.size(); i++) {
    if (is_contain(distinct_letters, user_name[i]) == 0) {
      distinct_letters.push_back(user_name[i]);
    }
  }

  /**
   * Decision Logic: The output depends on the parity of the number of
   * distinct characters.
   */
  if (distinct_letters.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}
