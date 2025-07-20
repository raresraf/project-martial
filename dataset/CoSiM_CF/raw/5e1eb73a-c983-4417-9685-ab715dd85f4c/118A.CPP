#include <iostream>
using namespace std;
int main() {
  string s, g = "";
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] >= 65 && s[i] <= 90) {
      s[i] = s[i] - 'A' + 'a';
    }
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'o' || s[i] == 'i' ||
        s[i] == 'u' || s[i] == 'y')
      continue;
    else
      g = g + "." + s[i];
  }
  cout << g << endl;
}