#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  if (s[0] >= 65 && s[0] <= 90) {
    cout << s;
    return 0;
  }
  s[0] = s[0] - 32;
  cout << s;
  return 0;
}
