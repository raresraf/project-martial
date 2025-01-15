// https://codeforces.com/problemset/problem/236/A
// 231A. Boy or Girl

#include <bits/stdc++.h>
using namespace std;

int countDistinct(string s) {
  unordered_map<char, int> m;
  for (int i = 0; i < s.length(); i++) {
    m[s[i]]++;
  }
  return m.size();
}
int main() {
  string str;
  cin >> str;
  int n = countDistinct(str);
  if (n & 1) {
    cout << "IGNORE HIM!";
  } else
    cout << "CHAT WITH HER!";
  return 0;
}
