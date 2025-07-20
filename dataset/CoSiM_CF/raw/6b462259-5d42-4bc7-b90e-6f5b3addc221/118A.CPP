#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     \
  int T;                                                                       \
  cin >> T;                                                                    \
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int
using namespace std;
int isVowel(char c) {
  if (c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'a' ||
      c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'Y' || c == 'y')
    return 1;
  else
    return 0;
}
int isUpper(char c) {
  if (c >= 65 && c <= 90)
    return 1;
  else
    return 0;
}
int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (isUpper(s[i])) {
      s[i] += 32;
    }
  }
  for (int i = 0; i < s.length(); i++) {
    if (!isVowel(s[i])) {
      cout << "." << s[i];
    }
  }
}
