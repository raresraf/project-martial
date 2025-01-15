#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  string s;
  cin >> s;
  set<char> se;
  int ris = 0;
  for (int i = 0; i < s.length(); i++) {
    if (se.find(s[i]) == se.end()) {
      ris++;
      se.insert(s[i]);
    }
  }
  if (ris % 2)
    cout << "IGNORE HIM!"
         << "\n";
  else
    cout << "CHAT WITH HER!"
         << "\n";
  return 0;
}
