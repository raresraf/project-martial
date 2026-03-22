/**
 * @file 236A.CPP
 * @brief Determines user gender based on the number of distinct characters in a username.
 *
 * This program solves a problem where the gender of a user is decided by whether the
 * number of unique characters in their username is even or odd.
 * Even -> Female ("CHAT WITH HER!"), Odd -> Male ("IGNORE HIM!").
 *
 * @b Algorithm: The number of distinct characters is found by inserting all
 *   characters of the input string into a `std::set` and then checking the set's size.
 * @b Time_Complexity: O(L * log(U)), where L is the string length and U is the
 *   number of unique characters (at most 256). Each set insertion takes O(log U).
 * @b Space_Complexity: O(U) to store the unique characters in the set.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // A set to store the distinct characters from the input username.
  set<int> s;
  // d: stores the length of the string and later the count of unique characters.
  // c, n: declared but unused.
  int d, c = 0, n;
  // A C-style character array to hold the input username.
  char ch[1000];
  
  // Read the entire line of input into the character array.
  cin.getline(ch, 1000);
  // Get the length of the input string.
  d = strlen(ch);

  /**
   * @brief Insert all characters into a set to find the unique ones.
   * The `std::set` data structure automatically handles duplicates.
   */
  for (int i = 0; i < d; i++) {
    s.insert(ch[i]);
  }
  
  // Get the count of unique characters, which is the final size of the set.
  d = s.size();
  
  /**
   * @brief Check if the number of distinct characters is even or odd.
   * If even, the user is identified as female. Otherwise, male.
   */
  if (d % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
    
  return 0;
}
