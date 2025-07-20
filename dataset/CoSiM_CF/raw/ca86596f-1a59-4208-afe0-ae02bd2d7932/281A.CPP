#include <iostream>
#include <string>
using namespace std;
int main() {
  int i;
  string s;
  cin >> s;
  if (int(s[0]) > 90) {
    cout << char(int(s[0] - 32));
    for (i = 1; i < s.length(); i++)
      cout << s[i];
  } else
    cout << s;
  return 0;
}
