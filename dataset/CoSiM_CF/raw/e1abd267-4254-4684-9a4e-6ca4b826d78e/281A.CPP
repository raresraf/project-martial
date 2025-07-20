#include <bits/stdc++.h>
using namespace std;

int main(void) {
  string s;

  cin >> s;

  if (*(s.begin()) >= 'a' && *(s.begin()) <= 'z') {
    *(s.begin()) -= 32;
  }

  cout << s;
  return 0;
}
