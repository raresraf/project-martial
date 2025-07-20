#include <bits/stdc++.h>

using namespace std;

int main() {

  char a;

  while (cin >> a) {
    if (!strchr("AEIOUYaeiouy", a)) {
      cout << '.' << (char)tolower(a);
    }
  }
}