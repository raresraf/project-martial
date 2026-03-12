/**
 * @5bd4a5d3-740a-4137-90da-4c26ad392216/110A.CPP
 * @brief This program solves problem 110A, known as "Nearly Lucky Number".
 * A number is considered "lucky" if it contains only the digits 4 and 7.
 * A number is "nearly lucky" if the count of lucky digits within it (4s and 7s)
 * is itself a lucky number (i.e., the count is either 4 or 7).
 * The program determines if a given large integer 'n' is a nearly lucky number.
 *
 * Algorithm: Digit extraction and counting, followed by a check on the count.
 * Time Complexity: O(log10(N)) where N is the input number, as it processes each digit once.
 * Space Complexity: O(1) for storing variables.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream.
using namespace std;     // Uses the standard namespace.

int main() {
  long long int n, m, i = 0, j; // Declares 'n' (input number), 'm' (current digit),
                               // 'i' (count of lucky digits), 'j' (unused).
  cin >> n; // Reads the long integer 'n' from standard input.
  
  /**
   * Block Logic: Extracts digits from 'n' one by one and counts the occurrences of lucky digits (4 or 7).
   * Invariant: 'i' accumulates the count of lucky digits found so far.
   */
  while (n != 0) { // Loop continues as long as there are digits left in 'n'.
    m = n % 10;    // Inline: Extracts the last digit of 'n'.
    n = n / 10;    // Inline: Removes the last digit from 'n'.
    // Inline: Checks if the extracted digit 'm' is either 4 or 7.
    if (m == 4 || m == 7) 
      i++; // Increments the lucky digit counter 'i'.
  }

  /**
   * Block Logic: Checks if the count of lucky digits itself is a lucky number.
   * Pre-condition: 'i' holds the total count of lucky digits in the input 'n'.
   * Invariant: Prints "YES" if 'i' is 4 or 7, otherwise "NO".
   */
  if (i == 4 || i == 7) // If the count of lucky digits is 4 or 7.
    cout << "YES" << endl; // Prints "YES".
  else
    cout << "NO" << endl; // Prints "NO".

  return 0; // Indicates successful program termination.
}
