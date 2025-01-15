#include <iostream>
using namespace std;
#include <string>
typedef long long ll;
int main() {
  string s;
  cin >> s;
  ll i = 0;
  while (s[i] != '\0') {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = (char)(s[i] - 'A' + 'a');
    if (!(s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
          s[i] == 'u' || s[i] == 'y'))
      cout << '.' << s[i];
    i++;
  }
  cout << "\n";
}