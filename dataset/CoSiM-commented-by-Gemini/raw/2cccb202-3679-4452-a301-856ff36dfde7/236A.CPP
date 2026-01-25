#include <iostream>
#include <set>

using namespace std;

#define long long ll;

string s;

void solve() {
  set<char> diffChar;
  for (int i = 0; i < s.length(); i++) {
    if (diffChar.count(s[i]) == 0)
      diffChar.insert(s[i]);
  }
  string answer = diffChar.size() % 2 ? "IGNORE HIM!" : "CHAT WITH HER!";
  cout << answer;
}

int main() {
  cin >> s;
  solve();
}