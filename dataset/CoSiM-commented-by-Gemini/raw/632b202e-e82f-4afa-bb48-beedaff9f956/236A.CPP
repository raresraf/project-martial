/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem.
 *
 * @problem The program determines a user's gender based on their username. If the
 * number of distinct characters in the username is odd, the user is considered
 * male ("IGNORE HIM!"). If the number is even, the user is female ("CHAT WITH HER!").
 *
 * @algorithm The algorithm uses a `std::set` to efficiently count the number of
 * unique characters in the input string.
 * 1. Read the username string.
 * 2. Iterate through the string and insert each character into the `std::set`.
 *    The set's properties ensure that only unique characters are stored.
 * 3. After iterating, the size of the set is equal to the number of distinct characters.
 * 4. Check if the size is odd or even using the modulo operator (`% 2`).
 * 5. Print the corresponding message based on the result.
 *
 * @time_complexity O(N log K), where N is the length of the string and K is the
 * number of unique characters. Each of the N insertions into the set takes O(log K) time.
 * @space_complexity O(K) to store the unique characters in the set.
 */
#include <algorithm>
#include <iostream>
#include <set>

using namespace std;

int main() {

  string s;
  cin >> s;
  
  // A set is used to automatically store only the unique characters.
  set<char> nchars;
  int i;

  // Iterate through the input string and insert each character into the set.
  for (i = 0; i < s.size(); i++) {
    nchars.insert(s[i]);
  }

  // Check if the number of distinct characters is odd or even.
  if (nchars.size() % 2 != 0)
    cout << "IGNORE HIM!" << endl;
  else
    cout << "CHAT WITH HER!" << endl;

  return 0;
}
