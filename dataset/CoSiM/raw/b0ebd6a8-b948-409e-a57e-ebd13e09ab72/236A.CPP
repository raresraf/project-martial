// https://codeforces.com/problemset/problem/236/A

#include <bits/stdc++.h>

using namespace std;

string solve(const string &s) {
  unordered_set<char> s_set(s.begin(), s.end());
  if (s_set.size() & 1) {
    return "IGNORE HIM!";
  } else {
    return "CHAT WITH HER!";
  }
}

int main() {
  string s;
  cin >> s;
  cout << solve(s) << endl;
}
