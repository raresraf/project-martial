#include <iostream>

using namespace std;

int main() {

  string s;
  int i, n;
  cin >> n >> s;
  int c = 0;

  for (i = 0; i < n - 1; i++) {
    if (s[i] == s[i + 1])
      c++;
  }
  cout << c << endl;
}