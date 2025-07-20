//
// Created by Conor on 5/27/2020.
//

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
using namespace std;

int change_case(char c) {
  if (isupper(c))
    return tolower(c);
  return c;
}

int main() {
  string a, b;
  cin >> a;
  cin >> b;

  transform(a.begin(), a.end(), a.begin(), change_case);
  transform(b.begin(), b.end(), b.begin(), change_case);

  if (a == b) {
    cout << 0;
  } else if (a.compare(b) > 0) {
    cout << 1;
  } else {
    cout << -1;
  }

  return 0;
}