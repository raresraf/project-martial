#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  unordered_set<char> sc;
  for (int i = 0; i < s.length(); i++)
    sc.insert(s[i]);
  if (sc.size() % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}