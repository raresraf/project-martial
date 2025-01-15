#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  char prev = s[0];
  int ans = 0;
  for (int i = 1; i < s.size(); i++) {
    if (s[i] == prev) {
      ans++;
    }
    prev = s[i];
  }
  cout << ans;
  return 0;
}