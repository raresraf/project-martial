/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem from Codeforces.
 * @details The problem defines a "lucky number" as one containing only digits
 * 4 and 7. A number is "nearly lucky" if the count of its lucky digits is
 * itself a lucky number.
 *
 * @algorithm
 * 1. Read the input number as a string to allow for digit-by-digit iteration.
 * 2. Initialize a counter `cnt` to 0.
 * 3. Loop through each character of the string. If a character is '4' or '7',
 *    increment `cnt`.
 * 4. After the loop, the total count of lucky digits is known. The problem
 *    then requires checking if this count is a lucky number. Given the
 *    constraints on the input length, the only possible lucky numbers the
 *    count can be are 4 or 7.
 * 5. Check if `cnt` is equal to 4 or 7.
 * 6. Print "YES" if it is, and "NO" otherwise.
 *
 * @complexity
 * Time Complexity: O(L), where L is the number of digits in the input number,
 * due to the single pass through the string.
 * Space Complexity: O(L) to store the number as a string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s; // To store the input number as a string.
  cin >> s;
  
  int cnt = 0; // Counter for the number of lucky digits ('4' or '7').
  
  /**
   * @block
   * @brief Iterate through the string to count the occurrences of lucky digits.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '7' || s[i] == '4')
      cnt++;
  }
  
  /**
   * @brief Check if the count of lucky digits is itself a "lucky number" (4 or 7).
   */
  if (cnt == 4 || cnt == 7)
    cout << "YES" << endl;
  else
    cout << "NO";
    
  return 0;
}
