#include <bits/stdc++.h>
using namespace std;

int countUnique(string str) {
  sort(str.begin(), str.end());
  auto i = unique(str.begin(), str.end());
  return i - str.begin();
}

int main() {
  string str;
  cin >> str;
  if (countUnique(str) % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
