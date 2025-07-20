#include <bits/stdc++.h>
#include <unordered_map>
using namespace std;

int main() {
  string s;
  getline(cin, s);

  unordered_map<int, char> m;

  for (int i = 0; i < s.length(); i++)
    m[s[i]]++;

  if (m.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
