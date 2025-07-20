#include <iostream>
#include <string>

using namespace std;

int main() {
  string s, ans = "";
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    char lower = tolower(s[i]);
    if (lower != 'u' && lower != 'o' && lower != 'e' && lower != 'i' &&
        lower != 'a' && lower != 'y') {
      ans += '.';
      ans += lower;
    }
  }
  cout << ans;
}