// CodeForces
// Problem 96A [Football]

#include <iostream>
using namespace std;

int main() {
  int checker = 0, counter = 0;

  string str;
  cin >> str;
  int l = str.length();

  for (int i = 1; i < l; i++) {
    if (str[i] == str[i - 1]) {
      counter++;
    } else {
      counter = 0;
    }

    if (counter == 6) {
      checker = 1;
      cout << "YES";
      break;
    }
  }

  if (checker == 0) {
    cout << "NO";
  }
  return 0;
}