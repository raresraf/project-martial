#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  string input = "";
  string res = "";
  set<char> xSet;
  cin >> input;
  for (int i = 0; i < input.length(); i++) {

    xSet.insert(input[i]);
  }
  if (xSet.size() % 2 == 0) {
    res = "CHAT WITH HER!";
  } else {
    res = "IGNORE HIM!";
  }
  cout << res << endl;
  return 0;
}
