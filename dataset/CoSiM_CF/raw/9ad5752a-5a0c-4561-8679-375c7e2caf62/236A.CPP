#include <bits/stdc++.h>
using namespace std;

int main() {
  set<char> s;
  string name;
  cin >> name;

  for (int i = 0; i < name.size(); i++) {
    s.insert(name[i]);
  }

  if (s.size() % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }

  return 0;
}
