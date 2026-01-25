#include <bits/stdc++.h>
using namespace std;

int main() {
  int testcases, x = 0;
  string input;
  cin >> testcases;
  for (int i = 0; i < testcases; i++) {
    cin >> input;
    if (input[1] == '+') {
      x++;
    } else {
      x--;
    }
  }
  cout << x;

  return 0;
}
