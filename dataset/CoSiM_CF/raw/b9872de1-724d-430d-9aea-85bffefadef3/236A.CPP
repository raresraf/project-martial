#include <bits/stdc++.h>

using namespace std;

int main() {
  string name;
  cin >> name;
  set<char> repeats;
  for (int i = 0; i < name.size(); i++) {
    repeats.insert(name[i]);
  }

  if (repeats.size() % 2 == 1) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
}
