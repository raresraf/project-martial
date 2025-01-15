// CodeForces
// Problem 58A [Chat Room]

#include <bits/stdc++.h>
using namespace std;

int main() {
  string hello = "hello";

  string inputstr;
  cin >> inputstr;

  unsigned int strindex, checker = 0;
  for (strindex = 0; strindex < inputstr.length(); strindex++) {
    if (inputstr[strindex] == hello[checker]) {
      checker++;
    }
  }
  if (checker >= 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}