#include <algorithm>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ull unsigned long long
#define ll long long

using namespace std;

int main() {
  string s;
  cin >> s;
  string ans = "";
  for (int i = 0; i < s.size(); i++) {
    if (s[i] < 'a') {
      s[i] = (char)(s[i] + 'a' - 'A');
    }

    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      continue;
    } else {
      ans += '.';
      ans += s[i];
    }
  }
  cout << ans;
  return 0;
}