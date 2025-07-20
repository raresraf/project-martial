#include <iostream>
#include <string>
using namespace std;
int main() {
  int t;
  cin >> t;
  for (int i = 0; i < t; i++) {
    string s;
    cin >> s;
    if (s.length() > 10)
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      cout << s << endl;
  }
}
