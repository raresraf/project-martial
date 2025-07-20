#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#define For(i, n) for (i = 0; i < n; i++)
#define ll long longf
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)

using namespace std;

int main() {
  string s;
  cin >> s;

  int length = s.length() / 2 + 1;

  char s1[length];

  int j = 0;
  for (int i = 0; i < s.length(); i = i + 2) {
    s1[j] = s[i];
    j++;
  }

  sort(s1, s1 + length);

  for (int i = 0; i < length - 1; i++) {
    cout << s1[i] << "+";
  }
  cout << s1[length - 1];

  return 0;
}
