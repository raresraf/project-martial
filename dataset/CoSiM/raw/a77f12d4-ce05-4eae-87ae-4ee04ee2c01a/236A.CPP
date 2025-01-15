#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  set<int> n;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    n.insert(s[i]);
  }
  if (n.size() % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}
