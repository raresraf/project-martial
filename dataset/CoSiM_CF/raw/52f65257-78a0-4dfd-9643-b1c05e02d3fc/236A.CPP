#include <bits/stdc++.h>
using namespace std;

int main() {
  set<char> inp;
  string input;
  cin >> input;
  int len = input.length();
  for (int i = 0; i < len; i++) {
    inp.insert(input[i]);
  }

  if (inp.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
}
