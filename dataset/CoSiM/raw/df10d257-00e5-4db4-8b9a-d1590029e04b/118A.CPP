#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>

using namespace std;

typedef long long ll;

int isvowel(int c) {
  c = tolower(c);
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

int main() {
  string s;
  cin >> s;
  for (auto c : s) {
    if (!isvowel(c)) {
      printf(".%c", tolower(c));
    }
  }
  cout << '\n';
}
