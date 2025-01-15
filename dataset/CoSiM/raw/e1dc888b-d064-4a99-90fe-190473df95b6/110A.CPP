#include <bits/stdc++.h>

using namespace std;

bool isLucky(int ld);

int main() {
  string input;
  cin >> input;
  int ld = 0; // It's a fuckin' must to initialize flags!!
  for (int i = 0; i < input.length(); i++) {
    if (input[i] == '4' || input[i] == '7') {
      ++ld;
    }
  }
  // cout << ld;
  if (isLucky(ld)) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}

bool isLucky(int ld) {
  int luckyFlag = 0;
  string ldString = to_string(ld); // Sexy method to convert a number to string
  for (int i = 0; i < ldString.length(); i++) {
    if (ldString[i] == '4' || ldString[i] == '7') {
      ++luckyFlag;
    }
  }
  if (luckyFlag < ldString.length()) {
    return false;
  } else {
    return true;
  }
}
