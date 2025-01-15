#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, i;
  string s;
  int x = 0;
  cin >> n;

  for (i = 0; i < n; i++) {
    cin >> s;
    if (s[1] == '+') {
      ++x;
    } else {
      --x;
    }
  }
  cout << x;

  return 0;
}