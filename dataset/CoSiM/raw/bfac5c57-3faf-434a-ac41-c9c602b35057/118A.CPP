#include <iostream>
#include <vector>

using namespace std;

char voice[20] = {'A', 'a', 'E', 'e', 'I', 'i', 'O', 'o', 'U', 'u', 'Y', 'y'};
vector<char> v;

int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); ++i) {
    bool b = false;
    for (int j = 0; j < 12; ++j)
      if (s[i] == voice[j])
        b = true;
    if (b == false) {
      cout << '.';
      if (s[i] >= 'A' && s[i] <= 'Z')
        cout << char('a' + s[i] - 'A');
      else
        cout << s[i];
    }
  }
}