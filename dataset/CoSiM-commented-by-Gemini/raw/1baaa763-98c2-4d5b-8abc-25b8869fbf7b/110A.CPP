/**
 * @file 110A.CPP
 * @brief Solution for a competitive programming problem, "Nearly Lucky Number".
 *
 * This program checks if a given number is "nearly lucky". A number is defined
 * as nearly lucky if the count of its "lucky digits" ('4' and '7') results
 * in a number that is itself lucky (i.e., the count is 4 or 7). This version
 * processes the input as a string and iterates through its characters.
 *
 * @algorithm
 * 1. Read the input number as a string.
 * 2. Initialize a counter to zero.
 * 3. Iterate through each character of the string.
 * 4. If a character is '4' or '7', increment the counter.
 * 5. After the loop, check if the final count is equal to 4 or 7.
 * 6. Print "YES" if the condition is met, otherwise print "NO".
 *
 * @time_complexity O(L), where L is the length of the input string (number of digits).
 * @space_complexity O(L) to store the input string.
 */

#include <iostream>

using namespace std;

int main() {
  string n;
  // 'sum' acts as a counter for the number of lucky digits found.
  int i, sum = 0;

  cin >> n;

  // Block Logic: Iterate through each character of the input string to count
  // the occurrences of lucky digits.
  for (i = 0; i < n.size(); i++) {
    if (n[i] == '4' || n[i] == '7')
      sum++;
  }

  // Block Logic: Check if the total count of lucky digits is a lucky number itself.
  // Pre-condition: 'sum' holds the total count of '4's and '7's.
  if (sum == 4 || sum == 7)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
