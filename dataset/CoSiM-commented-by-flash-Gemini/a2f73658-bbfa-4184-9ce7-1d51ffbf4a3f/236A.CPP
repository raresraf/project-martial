/**
 * @file 236A.CPP
 * @brief Solution to the "Boy or Girl" competitive programming problem (Codeforces 236A).
 * The problem determines a user's gender based on their username. If the number
 * of distinct characters in the username is odd, the user is a "boy"; if it's
 * even, the user is a "girl". The program outputs "IGNORE HIM!" for boys and
 * "CHAT WITH HER!" for girls.
 * Algorithm: Count distinct characters using a set, then check parity.
 * Time Complexity: O(L log L) due to set insertion (or O(L) if using a hash set or boolean array for char presence),
 * where L is the length of the input username.
 * Space Complexity: O(alphabet_size) for the set (or O(1) if considering fixed alphabet size).
 */
#include <algorithm> // Included but not directly used in this specific implementation.
#include <iostream>  // For input/output operations.
#include <set>       // For storing unique characters.
#include <vector>    // Included but not directly used in this specific implementation.

using namespace std;

int main(int argc, char const *argv[]) {
  set<char> s; // Declares a set to store unique characters from the input username.
  char c;      // Declares 'c' to store each character read from input.
  
  /**
   * Block Logic: Reads characters from standard input one by one until a newline character is encountered.
   * Invariant: At the start of each iteration, 'c' holds the last character read,
   * and 's' contains all unique characters encountered so far before 'c'.
   * Pre-condition: Input stream contains a username followed by a newline.
   * Post-condition: 's' contains all distinct characters from the username.
   */
  while (true) {
    scanf("%c", &c); // Reads a single character.
    // Conditional Logic: Breaks the loop if a newline character is read.
    if (c == '\n')
      break;

    s.insert(c); // Inserts the character into the set; duplicates are automatically handled.
  }
  
  /**
   * Conditional Logic: Checks the parity of the number of distinct characters.
   * If the size of the set (number of unique characters) is even, it's a "girl".
   * If it's odd, it's a "boy".
   */
  if (s.size() % 2 == 0)
    cout << "CHAT WITH HER!" << endl; // Prints message for a girl.
  else
    cout << "IGNORE HIM!" << endl;  // Prints message for a boy.
  
  return 0; // Indicates successful program execution.
}
