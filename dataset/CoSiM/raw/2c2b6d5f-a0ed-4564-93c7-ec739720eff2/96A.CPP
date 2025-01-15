#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  char x = s[0];
  int cu = 1;
  for (int i = 1; i < s.length(); i++) {
    if (s[i] == x)
      cu += 1;
    else {
      x = s[i];
      cu = 1;
    }
    if (cu >= 7) {
      break;
    }
  }
  if (cu >= 7)
    cout << "YES\n";
  else
    cout << "NO\n";
}