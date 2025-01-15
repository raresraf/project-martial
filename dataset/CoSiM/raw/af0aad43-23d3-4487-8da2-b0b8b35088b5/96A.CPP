#include <iostream>
#include <string>

using namespace std;

int main() {

  int i;
  string s;
  int c = 1;
  cin >> s;

  for (i = 0; i < s.size() - 1; i++) {

    if (s[i] == s[i + 1])
      c++;
    else
      c = 1;

    if (c >= 7) {
      cout << "YES";
      return 0;
    }
  }

  cout << "NO";
}