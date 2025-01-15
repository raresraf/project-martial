#include <bits/stdc++.h>
using namespace std;

int main() {
  int x = 0, n;
  string input;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> input;
    if (input[0] == '+' || input[1] == '+') {
      x++;
    } else {
      x--;
    }
  }
  cout << x << endl;
  return 0;
}
