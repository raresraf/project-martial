#include <bits/stdc++.h>
using namespace std;
int main() {
  set<char> store;
  string s;
  while (cin >> s) {
    for (int i = 0; i < s.size(); i++) {
      store.insert(s[i]);
    }
    if (store.size() % 2 != 0)
      cout << "IGNORE HIM!" << endl;
    else
      cout << "CHAT WITH HER!" << endl;

    store.clear();
  }
  return 0;
}
