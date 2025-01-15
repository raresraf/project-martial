#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'y' && s[i] != 'u')
      cout << "." << s[i];
  }
  return 0;
}
