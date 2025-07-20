#include <bits/stdc++.h>
using namespace std;
int main() {
  set<char> s;
  string st;
  cin >> st;
  for (int i = 0; i < st.length(); i++) {
    s.insert(st[i]);
  }
  if (s.size() % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}