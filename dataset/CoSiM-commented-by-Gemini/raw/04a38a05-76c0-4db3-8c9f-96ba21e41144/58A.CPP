/**
 * @file 58A.CPP
 * @brief Solves the "Chat room" problem from Codeforces contest 58.
 *
 * @details
 * The problem is to determine if it's possible to spell the word "hello" by deleting zero or more
 * characters from a given input string. This is a classic subsequence problem. We need to check
 * if "hello" is a subsequence of the input string.
 *
 * The program iterates through the input string while keeping track of the current character
 * it is looking for from the word "hello". It uses a pointer `j` for the target word "hello".
 * When a character in the input string matches the character `hello[j]`, the pointer `j` is
 * incremented. If `j` reaches 5, it means all characters of "hello" have been found in the
 * correct order, and the program can conclude that "hello" is a subsequence.
 *
 * Algorithm:
 * 1. Define the target word "hello".
 * 2. Initialize a pointer `j` to 0, representing the current character to find in "hello".
 * 3. Read the input string `s`.
 * 4. Iterate through each character of the input string `s` from left to right.
 * 5. In the loop, if the current character of `s` matches the `j`-th character of "hello",
 *    increment `j`.
 * 6. If `j` becomes 5, it means all characters of "hello" have been found in order, so we can
 *    stop early.
 * 7. After the loop, if `j` is 5, print "YES"; otherwise, print "NO".
 *
 * Time Complexity: O(L), where `L` is the length of the input string, as we do a single pass
 * through the string.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  // Read the input string from the user.
  cin >> s;
  // The target word to find as a subsequence.
  char hi[] = {'h', 'e', 'l', 'l', 'o'};

  int j = 0; // Pointer for the current character in the `hi` array.
  // Block Logic: Iterate through the input string to find the "hello" subsequence.
  // Invariant: 'j' holds the number of characters from "hello" that have been matched so far in the correct sequence.
  for (int i = 0; i < s.size(); ++i) {
    // Pre-condition: Check if the current character in the input string matches the character we're looking for in "hello".
    if (s[i] == hi[j])
      // If it matches, advance the pointer for the "hello" word.
      j++;
    // Optimization: If all characters of "hello" have been found, we can exit the loop early.
    if (j >= 5)
      break;
  }

  // After checking the entire string, if `j` is 5, it means "hello" was found as a subsequence.
  (j >= 5) ? cout << "YES\n" : cout << "NO\n";

  return 0;
}

// https://codeforces.com/problemset/problem/58/A
