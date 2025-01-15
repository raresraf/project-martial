// By archit-dev, contest: Codeforces Round #146 (Div. 2), problem: (A) Boy or
// Girl, Accepted, #, Copy
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
using namespace std;

bool search(string s, char a) {
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == a) {
      return true;
    }
  }
  return false;
}

int main() {
  string name;
  string name2("");
  cin >> name;
  for (int i = 0; i < name.size(); i++) {
    if (!search(name2, name[i])) {
      name2 = name2 + name[i];
    }
  }
  if (name2.size() % 2 == 0) {
    cout << "CHAT WITH HER!\n";
  } else {
    cout << "IGNORE HIM!\n";
  }
  return 0;
}
