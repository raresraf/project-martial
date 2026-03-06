/**
 * @file 236A.CPP
 * @brief Determines a user's gender based on the number of distinct characters in their username.
 *
 * This program reads a username and counts the number of unique characters. If the
 * count is even, the user is classified as female ("CHAT WITH HER!"). If the count
 * is odd, the user is classified as male ("IGNORE HIM!").
 *
 * @algorithm Hashing/Frequency Counting using a std::set.
 * @complexity
 * Time: O(N log N), where N is the length of the input string. Each insertion into
 *       a std::set takes O(log K) time, where K is the current size of the set.
 * Space: O(K), where K is the number of distinct characters in the string, to
 *        store the elements in the set. K is at most N, and for ASCII, at most 128.
 */
#include <iostream>
#include <string>
#include <set>
#include <cstdio> // For printf

// Using the standard namespace for competitive programming convenience.
using namespace std;

/**
 * @brief Main function to handle input and determine the output.
 */
int main() {
  string s;
  cin >> s;
  
  // A std::set automatically stores only unique elements.
  set<char> st;
  
  int i, len = s.length();
  
  // Pre-condition: 'st' is an empty set.
  // Block Logic: Iterate through the string and insert each character into the set.
  // Duplicates will be automatically ignored by the set.
  for (i = 0; i < len; i++) {
    st.insert(s[i]);
  }
  
  // Get the number of distinct characters by checking the size of the set.
  int a = st.size();
  
  // Invariant: 'a' holds the count of unique characters.
  // Block Logic: Check if the count is even or odd to determine the output.
  if (a % 2 == 0)
    printf("CHAT WITH HER!
");
  else
    printf("IGNORE HIM!
");
    
  return 0;
}
