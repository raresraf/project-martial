#include <cstring>
#include <iostream>

using namespace std;

bool v(char a) {
  return (a == 'a' || a == 'e' || a == 'i' || a == 'o' || a == 'u' || a == 'y');
}

int main() {
  string s;

  cin >> s;

  for (int i = 0; i < s.length(); i++) {
    s[i] = tolower(s[i]);
  }

  for (int i = 0; i < s.length(); i++) {
    if (!v(s[i])) {
      cout << "." << s[i];
    }
  }

  cout << endl;

  return 0;
}
