/**
 * @file 96A.CPP
 * @brief Determines if a given input string contains a substring of 7 or more identical consecutive characters.
 * This is a common pattern in competitive programming, often referred to as a "dangerous situation" check.
 * Algorithm: Linear scan with a counter for consecutive identical characters.
 * Time Complexity: O(N), where N is the length of the input string, as it involves a single pass.
 * Space Complexity: O(1) for auxiliary variables.
 */
///       From dust I came and to dust I shall return
/**
                    Asadullah Bin Rahman
                          CSE-17
    HAJEE MOHAMMAD DANESH SCIENCE AND TECHNOLOGY UNIVERSITY
*/
#include <bits/stdc++.h> // Common competitive programming header, includes most standard libraries.
#define sf(f) scanf("%d", &f)
#define pf(f) printf("%d\n", f)
#define PI 3.1415926535897932385
#define size 3000000
#define mp make_pair
#define pb push_back
#define one(x) __builtin_popcount(x)     /// int
#define onel(x) __builtin_popcountl(x)   /// long int
#define onell(x) __builtin_popcountll(x) /// long long
using namespace std;
typedef long long ll;

/// int dx[]= {-1,-1,0,0,1,1};
/// int dy[]= {-1,0,-1,1,0,1};
/// int dx[]= {0,0,1,-1};                 /*4 side move*/
/// int dy[]= {-1,1,0,0};                 /*4 side move*/
/// int dx[]= {1,1,0,-1,-1,-1,0,1};       /*8 side move*/
/// int dy[]= {0,1,1,1,0,-1,-1,-1};       /*8 side move*/
/// int dx[]={1,1,2,2,-1,-1,-2,-2};       /*knight move*/
/// int dy[]={2,-2,1,-1,2,-2,1,-1};       /*knight move*/
/// Always Remember   'A'=65,'Z'=90 'a'=97 'z'=122 '0'=48

int main() {
  string s; // Declares a string variable 's' to store the input.
  cin >> s; // Reads the input string from standard input.
  int count = 1; // Initializes a counter to track consecutive identical characters.
  int l = s.length(); // Stores the length of the input string for loop control.
  /**
   * Block Logic: Iterates through the input string to find sequences of identical characters.
   * Compares each character with its preceding character to identify consecutive matches.
   * Invariant: 'count' always stores the number of consecutive identical characters ending at the current index 'i'.
   */
  for (int i = 1; i < l; ++i) {
    // Conditional Logic: Checks if the current character is the same as the previous one.
    if (s[i] == s[i - 1]) {
      // cout<<"fuck\n"; // Debugging statement, commented out.
      count += 1; // Increments the counter for consecutive identical characters.
      // Inline: If 7 consecutive identical characters are found, it's a "dangerous situation".
      if (count == 7) {
        cout << "YES" << endl; // Prints "YES" and terminates the program.
        return 0;
      }
    }
    // Else Block: Resets the counter if the current character is different from the previous one.
    else
      count = 1;
  }

  cout << "NO" << endl; // If no sequence of 7 identical characters is found after checking the entire string, prints "NO".
  return 0;
}
