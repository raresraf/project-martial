#include <iostream>
using namespace std;
int main() {
  int t;
  cin >> t;
  int x = 0;
  for (int i = 0; i < t; i++) {
    string s;
    cin >> s;
    if ((s[0] == '+' && s[1] == '+') ||
        (s[s.length() - 1] == '+' && s[s.length() - 2] == '+'))
      x = x + 1;
    if ((s[0] == '-' && s[1] == '-') ||
        (s[s.length() - 1] == '-' && s[s.length() - 2] == '-'))
      x = x - 1;
  }
  cout << x << endl;
}
