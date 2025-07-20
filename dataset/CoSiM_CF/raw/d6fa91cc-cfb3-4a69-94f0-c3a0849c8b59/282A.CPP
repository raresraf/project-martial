#include <iostream>

using namespace std;

int main() {
  int n, x = 0, i;
  string s;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> s;
    if (s[1] == '-') {
      x--;
    } else {
      x++;
    }
    s = "";
  }
  cout << x;
  return 0;
}
