//
// Created by Conor on 5/28/2020.
//

#include <iostream>
#include <string>

using namespace std;

bool is_lucky(const string &n) {
  int count = 0;

  for (char i : n) {
    if (i == '4' || i == '7') {
      count++;
    }
  }

  if (count == 0) {
    return false;
  } else {
    int c = 0;
    for (char i : to_string(count)) {
      if (i == '4' || i == '7') {
        c++;
      }
    }
    return to_string(count).size() == c;
  }
}

int main() {
  string n;
  cin >> n;

  if (is_lucky(n)) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}