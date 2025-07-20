#include <iostream>
#include <string.h>
using namespace std;
int main() {
  int n, x;
  cin >> n;
  string s;
  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s.length() <= 10)
      cout << s << endl;

    else {
      x = s.length();

      cout << s.at(0) << (x - 2) << s.at(x - 1) << endl;
    }
  }
  return 0;
}
