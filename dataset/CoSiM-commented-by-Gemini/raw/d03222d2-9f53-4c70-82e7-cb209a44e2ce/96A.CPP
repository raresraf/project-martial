#include <bits/stdc++.h>
using namespace std;

int main() {
  string input;
  cin >> input;

  int len = input.length();
  int counter = 1;
  int last = (int)input[0] - 48;
  for (int i = 1; i < len; i++) {
    if ((int)input[i] - 48 == 0) {
      if (last == 0)
        counter++;
      else {
        counter = 1;
        last = 0;
      }
    }
    if ((int)input[i] - 48 == 1) {
      if (last == 1)
        counter++;
      else {
        counter = 1;
        last = 1;
      }
    }
    if (counter == 7)
      break;
  }

  if (counter == 7)
    cout << "YES";
  else
    cout << "NO";
}
