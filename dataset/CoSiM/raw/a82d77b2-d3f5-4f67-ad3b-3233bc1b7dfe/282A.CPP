#include <iostream>
#include <string>
using namespace std;
int main() {
  int n = 0, x = 0;
  cin >> n;
  string s;
  while (n--) {
    cin >> s;
    if (s[1] == '+')
      ++x;
    else
      --x;
  }
  cout << x;
  return 0;
}
