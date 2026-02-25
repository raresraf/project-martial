/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A (Boy or Girl).
 * @author Monil Soni (sonil06), DA-IICT
 *
 * @details
 * This program determines a user's gender based on the number of distinct
 * characters in their username. If the count of distinct characters is even,
 * the user is identified as female ("CHAT WITH HER!"). If the count is odd,
 * the user is identified as male ("IGNORE HIM!").
 *
 * @section algorithm
 * The core of the algorithm is to count the number of unique characters in the
 * input string.
 * 1. Read the username into a `std::string`.
 * 2. A `std::set<char>` is used to store the characters of the string. The set
 *    automatically handles uniqueness, so inserting all characters from the
 *    string into the set results in a collection of only the distinct characters.
 * 3. After populating the set, its size (`s.size()`) gives the count of
 *    distinct characters.
 * 4. The program checks if this count is even or odd using the modulo operator.
 * 5. It prints the corresponding message based on the result.
 *
 * Note: The code includes a `sort()` on the input string before inserting into
 * the set. This step is redundant as `std::set` inherently manages uniqueness
 * and order, making the initial sort unnecessary for the final count.
 *
 * @section complexity
 * - Time Complexity: O(L log L) due to the sort operation on the string of
 *   length L. If the sort were removed, the complexity would be O(L log K),
 *   where K is the number of unique characters (at most the size of the
 *   character set, e.g., 26 for lowercase English letters), because of the
 *   insertions into the set. Given the small alphabet size, this is effectively O(L).
 * - Space Complexity: O(K) to store the unique characters in the set.
 */

#include <bits/stdc++.h>

using namespace std;

// Various typedefs and macros common in competitive programming for brevity.
typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
#define __for(i, n) for (int i = 1; i <= n; i++)
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define ALL(x) x.begin(), x.end()

int main() {
  string str;
  cin >> str;

  // Sorting the string here is not necessary because std::set will handle
  // uniqueness regardless of the order of insertion.
  sort(str.begin(), str.end());

  int n = str.size();
  set<char> s;

  // Insert all characters from the string into a set to find the unique characters.
  for (int i = 0; i < n; i++) {
    s.insert(str.at(i));
  }

  // Check if the number of distinct characters is even or odd.
  if (s.size() % 2 == 0)
    // If even, the user is female.
    printf("CHAT WITH HER!\n");
  else
    // If odd, the user is male.
    printf("IGNORE HIM!\n");
    
  return 0;
}
