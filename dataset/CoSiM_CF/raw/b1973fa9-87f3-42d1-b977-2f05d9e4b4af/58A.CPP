// using a simple regex expression of
// *h*e*l*l*o* i.e any number of chars before each letter but their
// specific order will remain same

// Problem Type: Regex and String Matching

#include <bits/stdc++.h>
using namespace std;

bool checkPatternHolds(string s) {
  bool status = false;

  regex str1("(.*)h(.*)e(.*)l(.*)l(.*)o(.*)");

  if (regex_match(s, str1)) {
    status = true;
  } else {
    status = false;
  }
  return status;
}

int main() {
  string s;
  cin >> s;
  if (checkPatternHolds(s)) {
    cout << "YES"
         << "\n";
  } else {
    cout << "NO"
         << "\n";
  }
  return 0;
}