/**
 * @file 266A.CPP
 * @brief This program counts the number of adjacent identical characters in a given string.
 * This is a common pattern in competitive programming for problems involving string manipulation
 * and identifying redundant elements.
 *
 * Algorithm: Single pass iteration and comparison.
 * 1. Read the length of the string `n`.
 * 2. Read the input string `x`.
 * 3. Initialize a counter `cnt` to 0.
 * 4. Iterate through the string from the first character up to the second-to-last character.
 * 5. If a character is identical to its immediate successor, increment `cnt`.
 * 6. Print the final count `cnt`.
 * Time Complexity: O(N) where N is the length of the string, due to a single pass iteration.
 * Space Complexity: O(N) to store the input string, plus O(1) for the counter.
 */

/// Jos  Luiz da Silva Neto
/// Computer Engineering - Federal University of Itajub
#include <bits/stdc++.h>

using namespace std;

#define INF 0x3F3F3F3F
#define LINF 0x3F3F3F3F3F3F3F3FLL
#define pb push_back
#define mp make_pair
#define F first
#define S second

typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vii;
typedef long long ll;
typedef unsigned long long ull;

#define D(x) cout << #x << " = " << x << endl;

// Functional Utility: Arrays for directional movements (e.g., in grid problems), not used in this specific problem.
int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};

// Functional Utility: Global counter to store the number of identical adjacent characters.
int cnt;

/**
 * @brief Main function of the program.
 * Reads a string and counts adjacent identical characters.
 * @pre Input consists of an integer `n` (string length) and a string `x`.
 * @post Prints the total count of adjacent identical characters.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: Reads the length of the string.
  int n;
  cin >> n;
  // Functional Utility: Reads the input string.
  string x;
  cin >> x;
  // Block Logic: Iterates through the string to compare adjacent characters.
  // Invariant: `cnt` accurately stores the count of identical adjacent characters encountered up to `i-1`.
  // Pre-condition: `n - 1` ensures that `x[i+1]` is always a valid index.
  for (int i = 0; i < n - 1; i++)
    // Block Logic: Checks if the current character is identical to the next character.
    if (x[i] == x[i + 1])
      cnt++; // Functional Utility: Increments the counter if characters are identical.
  // Functional Utility: Prints the final count.
  printf("%d
", cnt);
  return 0;
}
