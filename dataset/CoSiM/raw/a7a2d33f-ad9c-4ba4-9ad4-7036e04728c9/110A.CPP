#include <iostream>
#include <string>
using namespace std;
int main() {
  int i, c(0);
  string s;
  cin >> s;
  for (i = 0; i < s.length(); i++) {
    if (s[i] == '4' || s[i] == '7')
      c++;
  }
  if (c == 4 || c == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
